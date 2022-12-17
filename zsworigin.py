from matplotlib import pyplot as plt


class Equation:

    def __init__(self, vc, vd):
        self.vc = vc
        self.vd = vd
        return

    def equation_solver_euler(self, x0, b, y, h):
        n = int((b - x0) / h)
        m = len(y)
        lgx = []
        lgy = []
        k1 = []
        for k in range(0, 20):
            k1.append(0)
        for i in range(0, n + 1):
            lgx.append(round(x0, 3))
            lgy.append(y[0])

            for j in range(0, m):
                k1[j] = h * self.rk_functions(x0, y)[j]
            for j in range(0, m):
                y[j] += k1[j]
            x0 += h

        return lgx, lgy

    def equation_solver_rungekutta(self, x0, b, y, h):
        n = int((b - x0) / h)
        m = len(y)
        lgx = []
        lgy = []
        k1 = []
        k2 = []
        k3 = []
        k4 = []
        for _ in range(0, 30):
            k1.append(0)
            k2.append(0)
            k3.append(0)
            k4.append(0)
        for i in range(0, n + 1):
            lgx.append(round(x0, 3))
            lgy.append(y[0])

            for j in range(0, m):
                k1[j] = h * self.rk_functions(x0, y)[j]
                k2[j] = h * self.rk_functions(x0 + 0.5 * h, holder(y, k1, 0.5))[j]
                k3[j] = h * self.rk_functions(x0 + 0.5 * h, holder(y, k2, 0.5))[j]
                k4[j] = h * self.rk_functions(x0 + h, holder(y, k3, 1))[j]
                

            for j in range(0, m):
                y[j] += (1. / 6.) * (k1[j] + 2 * (k2[j] + k3[j]) + k4[j])
            x0 += h

        return lgx, lgy

    def function(self, y, t):
        f = 0
        n = len(self.vc)
        for i in range(0, n - 1):
            if i == 0:
                f += (self.vc[0] / self.vc[n - 1]) * (t ** (self.vd[0] - self.vd[n - 1]))
            else:
                f += (self.vc[i] / self.vc[n - 1]) * (t ** (self.vd[i] - self.vd[n - 1])) * y[i - 1]
        return f

    def rk_functions(self, t, y):
        ddx = []
        n = len(y)
        for i in range(0, n - 1):
            ddx.append(y[i + 1])
        ddx.append((-1) * self.function(y, t))
        return ddx

   
def holder(vx, vk, scale):
    temp = [vx[0]]
    for i in range(1, len(vx)):
        temp.append(vx[i] + scale * vk[i])
    return temp


def painter(xp, yp):
    fig, ax = plt.subplots()
    fig.set_figwidth(18)
    fig.set_figheight(18)
    ax.plot(xp, yp)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid()
    ax.set_title("ODE")
    plt.show()

eq = Equation([1, 1, -1, 1], [1, 1, 0, 0])

xp, yp = eq.equation_solver_rungekutta(0, 10, [1, 1], 0.01)
painter(xp, yp)
