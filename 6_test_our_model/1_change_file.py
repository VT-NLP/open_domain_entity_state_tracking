import json

fw = open("prediction_format_define_wo_ESC.jsonl", "w",encoding="utf-8")

with open("prediction_format_wo_ESC.jsonl","r",encoding="utf-8") as f:
    for line in f:
        input_json = json.loads(line)
        # id_f = input_json["id"]
        # print(input_json)
        # break
        # Dic_={}
        # Dic_id=input_json["id"]
        # Dic_["question"] = input_json["question"]#+instruction+prompt
        # S=[x.strip() for x in input_json["answers"][0].replace("<pad>","").replace("</s>","").replace("<attr> ","").replace("</attr>","").
        #
        #     replace("<entity>","").replace("</entity>","").strip().split("[SN]")]
        S = [x.strip() for x in
             input_json["answers"][0].replace("<pad> ", "").replace("</s>", "").replace("[END]","").strip().split("[SN]")]
        # print(S)
        if S[-1]==".":
            input_json["answers"] =S[:-1]
        else:
            input_json["answers"] = S

        # input_json["answers"] =S[1:]
        fw.writelines(json.dumps(input_json) + "\n")
        fw.flush()
