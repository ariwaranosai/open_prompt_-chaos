import random

"""
label,text_a
example:
1,非常快，态度好。
0,不要了还显示送达？
"""
with open("waimai_10k.csv") as f:
    f.readline() # skip title
    total_data = []
    for line in f:
        total_data.append(line)
    random.shuffle(total_data)
    with open("waimai_10k_train.csv", "w") as g, open("waimai_10k_validation.txt", "w") as h:
        for line in total_data:
            if random.random() < 0.8:
                g.write(line)
            else:
                h.write(line)


