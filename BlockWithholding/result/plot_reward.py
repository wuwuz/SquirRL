import numpy as np
import matplotlib.pyplot as plt
import argparse

CLI = argparse.ArgumentParser()
CLI.add_argument(
    '--input',
    nargs='*',
    type=str,
    default = ""
)

CLI.add_argument(
    '--output',
    nargs='*',
    type=str,
    default = ""
)

CLI.add_argument(
    '--x',
    nargs='*',
    type=str,
    default = ["alpha"]
)

CLI.add_argument(
    '--y',
    nargs='*',
    type=str,
    default= ["Reward"]
)

CLI.add_argument(
    '--plot',
    nargs = '*',
    type = str,
    default = ["RL1reward", "NE1reward", "RL2reward", "NE2reward"]
)

CLI.add_argument(
    '--info',
    nargs = '*',
    type = str,
    default = []
)

marker_dict = {
    "RL1reward" : 'o',
    "NE1reward" : 'v' ,
    "RL2reward" : 'P',
    "NE2reward" : '.',
}

label_dict = {
    "RL1reward" : 'RL-$r_1$',
    "NE1reward" : 'NE-$r_1$',
    "RL2reward" : 'RL-$r_2$',
    "NE2reward" : 'NE-$r_2$',
}


args = CLI.parse_args()
input_file = args.input[0]
if (args.output[0] == "same"):
    output_file = args.input[0].replace("csv", "pdf")
else:
    output_file = args.output[0]

fig, ax = plt.subplots(figsize=(12,9))

x_axis = args.x[0]
print(x_axis)
y_axis = args.y[0]
print(y_axis)

input = open(input_file, "r")
first_line = input.readline().strip()
name_list = first_line.split(',')
for i in range(len(name_list)): name_list[i] = name_list[i].strip()

data = np.zeros((10, 100), dtype=np.float)
i = 0

print(name_list)

line = input.readline()
while line:
    print(line)
    cur = line.split(',')
    for j in range(len(cur)) :
        data[j][i] = float(cur[j])
    i += 1
    line = input.readline()

n = i
x_axis_data = data[name_list.index(x_axis)][:n]

for i in range(len(args.plot)):
    try :
        if (args.plot[i] == "RL1reward"):
            ax.plot(x_axis_data, data[name_list.index(args.plot[i])][:n], label = label_dict[args.plot[i]], marker = marker_dict[args.plot[i]], markersize = 20, linestyle = ':', color = "black")
        else:
            ax.plot(x_axis_data, data[name_list.index(args.plot[i])][:n], label = label_dict[args.plot[i]], marker = marker_dict[args.plot[i]], markersize = 20, linestyle = ':')
    except :
        print("no data " + args.plot[i])


print(type(ax))
ax.legend(fontsize = 24, loc="lower left")
print(x_axis)
if (x_axis == "alpha"):
    ax.set_xlabel(r"$P_2$'s Hash Power $m_2$", fontsize = 28)
elif (x_axis == "dev"):
    ax.set_xlabel("Standard Deviation of Gaussian Process", fontsize = 28)
else:
    ax.set_xlabel(x_axis, fontsize=28)
ax.set_ylim(0.4, 1.1)
ax.set_ylabel(y_axis, fontsize = 28)
ax.tick_params(labelsize=22)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.tight_layout()

plt.savefig(output_file, format="pdf", pad_inches=0, bbox_inches='tight')
plt.show()
plt.clf()
exit()
