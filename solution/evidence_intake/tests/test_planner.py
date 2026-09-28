import copy
from solution.evidence_intake.planner import plan_inspection


def catalog():
    rows=[]
    for photo,families in [('a',[0,1]),('b',[0]),('c',[1]),('d',[2])]:
        for family in families:
            rows.append({'parent_id':photo,'region_id':f'{photo}-{family}', 'family_internal':family,
                         'source_relative_path':photo+'.jpg','source_sha256':photo*64,
                         'global_community':0,'global_context_name':'example','is_r3_core':False})
    return {'items':rows,'families':[{'family_internal':i} for i in range(3)]}


def test_replanning_counts_parent_once_and_skips_reviewed():
    data=catalog()
    data['items'].append(dict(data['items'][0],region_id='duplicate-region-same-photo'))
    first=plan_inspection(data,budget=1)
    assert first['items'][0]['canonical_id']=='a'
    second=plan_inspection(data,budget=3,reviewed=['a'])
    assert 'a' not in [i['canonical_id'] for i in second['items']]
    assert second['before']['families_repeated']==0
    assert second['after']['families_repeated']==2


def test_core_flags_and_target_labels_cannot_change_plan():
    data=catalog()
    other=copy.deepcopy(data)
    for row in other['items']:
        row.update(is_r3_core=True,target=1,expected_family_key='unrelated')
    assert plan_inspection(data,budget=3)==plan_inspection(other,budget=3)


def test_zero_weight_and_invalid_photo_are_handled():
    import pytest
    result=plan_inspection(catalog(),budget=4,weights={0:0,1:0,2:1})
    assert [i['canonical_id'] for i in result['items']]==['d']
    with pytest.raises(ValueError):plan_inspection(catalog(),reviewed=['missing'])
