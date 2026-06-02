import re

with open("tests/test_api.py", "r") as f:
    content = f.read()

expected_classes = '{"normal", "parkinsons", "stroke", "diabetic_neuropathy", "cerebellar_ataxia", "osteoarthritis", "dementia", "cerebral_hemorrhage", "cerebral_infarction", "disc_herniation", "rheumatoid_arthritis"}'

content = content.replace('assert body["final_prediction"] in {"normal", "antalgic", "ataxic", "parkinsonian"}', f'assert body["final_prediction"] in {expected_classes}')

with open("tests/test_api.py", "w") as f:
    f.write(content)
