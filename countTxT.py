path = "/home/djy/Desktop/222.txt"
sum = 0
len = 400
with open(path,"r",encoding="utf-8") as f:
    str = f.readline()
    list = str.split(",")
    for i in range(0,len):
        num_str = list[i]
        num_float = float(num_str)
        sum = sum + num_float
    mean = sum / len
    print(mean)