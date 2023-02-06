import json

def get_file_dict(file):
    DIc={}
    with open(file,"r",encoding="utf-8") as fr:
        for line in fr.readlines():
            dic_=json.loads(line)
            DIc[dic_["id"]]=dic_["answers"]
    return DIc

# groud_id_answer=get_file_dict("id_answers.jsonl")
s1=get_file_dict("prediction_format_define_wo_ESC.jsonl")
s2=get_file_dict("prediction_format_reward_define.jsonl")

lengths = []
fw=open("prediction_format_constraint_reward_define_0.1.jsonl","w",encoding="utf-8")
with open("prediction_format_define_wo_ESC.jsonl", encoding="utf-8") as f2:
    R_cal = []
    R = 0
    al = 0
    attr_R = 0
    for line in f2:
        al = al + 1
        # print("--------al", al)
        input_json = json.loads(line)
        id = input_json["id"]
        answers=input_json["answers"]
        tuples_new=[]
        for tuple in answers:
            # print(tuple)
            if "of" in tuple and "was" in tuple and "and" in tuple and "before" in tuple and "afterwards" in tuple:
                tuples_new.append(tuple)
        if len(tuples_new)==0:
            input_json["answers"] = ['There will be no change.']
        else:
            input_json["answers"] = list(tuples_new+s2[id])
        fw.writelines(json.dumps(input_json) + "\n")
        fw.flush()
