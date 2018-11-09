import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
fig = plt.figure(facecolor='#07000d')
# 272822 monokai
# 07000d purple black
fig.canvas.set_window_title('Training Loss (negative log likelihood)')
ax1 = fig.add_subplot(1, 1, 1, facecolor='#07000d')

# clears the file every time program is started
# pulldata = open('LossData.txt', 'w')
# pulldata.close()


def plotgraph():
    pulldata = open('LossData.txt', 'r').read()
    dataArray = pulldata.split('\n')
    xar = []
    yar = []
    for line in dataArray:
        if len(line) > 1:
            x, y = line.split(',')
            xar.append(int(x))
            yar.append(float(y))
    ax1.clear()
    plt.xlabel('Iteration(per 100)', color='#ABAA98')
    plt.ylabel('Smooth Loss', color='#ABAA98')
    ax1.tick_params(axis='y', colors='#ABAA98')
    ax1.tick_params(axis='x', colors='#ABAA98')
    ax1.plot(xar, yar, color='#66D9EF')
    ax1.spines['bottom'].set_color('#5998ff')
    ax1.spines['left'].set_color('#5998ff')
    ax1.spines['right'].set_color('#5998ff')
    ax1.spines['top'].set_color('#5998ff')
    ax1.grid(True, color='#ABAA98', alpha=0.2, linewidth='0.5')
    title_obj = plt.title('Training Loss (negative log likelihood)')
    plt.getp(title_obj)  # print out the properties of title
    plt.getp(title_obj, 'text')  # print out the 'text' property for title
    plt.setp(title_obj, color='#ABAA98')
    plt.show()


if __name__ == '__main__':

    plotgraph()
