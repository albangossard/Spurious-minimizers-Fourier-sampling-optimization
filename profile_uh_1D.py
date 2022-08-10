import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import recon, utils


list_P = [1,10,50,100,1000]
list_color = (plt.rcParams['axes.prop_cycle'].by_key()['color'])[:len(list_P)]

fig = plt.figure(0,figsize=(8,5))
ax = fig.add_subplot(1,1,1)

for e,(P,color) in enumerate(zip(list_P,list_color)):
    print(P)

    np.random.seed(1)
    N = 128

    f = utils.gen(P, N).T

    xi = np.linspace(-N/2,N/2,50*N)

    assert N == f.shape[0]
    uh = recon.A(xi, N).dot(f)
    F = np.mean(np.abs(uh)**2, axis=1)
    plt.semilogy(xi, F, label='P='+str(P), color=color)

    der = np.diff(F)
    ind = np.argwhere((der[:-1]>=0) & (der[1:]<0))
    y = 10**(e/len(list_P)-2)
    plt.scatter(xi[ind], np.ones_like(xi[ind])*y, color=color, marker='+', zorder=10)

plt.xlim(-N/2,N/2)
plt.ylim(1e-6,8e-1)

locx = plticker.MultipleLocator(base=1)
ax.xaxis.set_major_locator(locx)
nticks = len(ax.get_xticklabels())-3
ticks = ['']*len(ax.get_xticklabels())
ticks[1:-1] = [str(int(e-nticks/2.)) if e%5==(nticks/2)%5 else '' for e in range(nticks)]
ax.set_xticklabels(ticks)

plt.xlabel(r'$\xi$')
plt.ylabel(r'$\rho_P$')
plt.legend(loc=4)
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/uh_1D_L.pdf")
plt.show()
