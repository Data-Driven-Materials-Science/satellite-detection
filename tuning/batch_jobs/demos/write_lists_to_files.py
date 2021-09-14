file_name = "list_output_demo.txt"
for i in range(5):
    for j in range(5):
        s = [i, j, 13, 17]
        with open(file_name, "a") as output:
            output.write(str(s))
        f = open(file_name, "a")
        f.write('\n')
        f.close()