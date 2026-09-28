"""Portable CLS-only extraction matching the recorded XBAT DINOv2 protocol.

Requires a CUDA worker and legally obtained source archives and checkpoint.
This port preserves crop, letterbox, normalization, precision, order and pooling;
the archived study features were computed by the original extraction program.
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

WEIGHT_SHA = '55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['archives', 'manifest', 'checkpoint', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import torch
    import timm
    from PIL import Image, ImageOps
    from safetensors.torch import load_file
    from torchvision.transforms.functional import to_tensor, normalize
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() == WEIGHT_SHA
    if (a.output/'xbat_cls.npy').exists():
        raise FileExistsError('Output already contains a feature bank')
    if not torch.cuda.is_available():
        raise RuntimeError('Run on an authorized CUDA worker')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False,
                              num_classes=0, dynamic_img_size=True)
    model.load_state_dict(load_file(str(a.checkpoint)), strict=True)
    model.eval().to('cuda')
    rows = sorted(json.loads(a.manifest.read_text(encoding='utf-8')),
                  key=lambda r: (r['specimen_id'], r['modality']))
    assert len(rows) == 842
    a.output.mkdir(parents=True, exist_ok=True)
    vectors = np.lib.format.open_memmap(a.output/'xbat_cls.npy', 'w+', np.float16, (842,768))
    index = []
    for i, r in enumerate(rows):
        with zipfile.ZipFile(a.archives/r['archive']) as z:
            raw = z.read(r['member'])
        assert hashlib.sha256(raw).hexdigest() == r['sha256']
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
        im = im.crop(r['foreground_box_native_xyxy'])
        scale = 336/max(im.size)
        fitted = im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.BICUBIC)
        canvas = Image.new('RGB',(336,336),(128,128,128))
        canvas.paste(fitted,((336-fitted.width)//2,(336-fitted.height)//2))
        x = normalize(to_tensor(canvas),(.485,.456,.406),(.229,.224,.225))
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
            tokens = model.forward_features(x.unsqueeze(0).to('cuda'))
        assert tuple(tokens.shape) == (1,577,768)
        vectors[i] = torch.nn.functional.normalize(tokens.float()[:,0],dim=-1).cpu().numpy()[0].astype(np.float16)
        index.append({'id':r['specimen_id']+':'+r['modality'],'index':i})
    vectors.flush()
    metadata = {'xbat':index,'model':'timm/vit_base_patch14_dinov2.lvd142m',
                'weights_sha256':WEIGHT_SHA,'torch':torch.__version__,'timm':timm.__version__,
                'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                'preprocessing':'foreground crop; 336 letterbox gray128 bicubic; ImageNet normalization',
                'pooling':'L2-normalized CLS; float16 storage'}
    (a.output/'feature_index.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')

if __name__ == '__main__':
    main()
