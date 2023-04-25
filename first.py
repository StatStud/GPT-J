

context = """We're here live with Channel 9 News reporting an incident this morning on the Peterson Highway.
              Just after 9AM this morning, a mini van was spotted in a hit and run incident, leaving two injured
              and seven diseased. Suspects are"""

input_ids = tokenizer(context, return_tensors="pt").input_ids
gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)
