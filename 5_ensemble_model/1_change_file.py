import json

fw = open("prediction_format_reward_define.jsonl", "w",encoding="utf-8")

with open("prediction_format_reward.jsonl","r",encoding="utf-8") as f:
    for line in f:
        input_json = json.loads(line)
        S = [x.strip() for x in
             input_json["answers"][0].replace("<pad> ", "").replace("</s>", "").replace("[END]","").strip().split("[SN]")]
        if S[-1]==".":
            input_json["answers"] =S[:-1]
        else:
            input_json["answers"] = S

        # input_json["answers"] =S[1:]
        fw.writelines(json.dumps(input_json) + "\n")
        fw.flush()
