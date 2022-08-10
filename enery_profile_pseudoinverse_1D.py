import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from progress.bar import IncrementalBar
import recon, utils


# Initialization variables
N = 16 # Size of the image
M = 2 # Number of sampling points


id=3
save=True

fontsize=12

xx = np.linspace(-N/2,N/2,N,endpoint=False)
np.random.seed(1+1)
u = np.random.randn(N)
if id==1:
    v=((-1)**(np.arange(N)-N/2)+1)/2
    plt.plot(v); plt.show()
    u = np.fft.fft(v)/np.sqrt(N)
    epsG = 0
    epsF = 0
    epsJ = 0
elif id==2:
    u = np.zeros(N); u[int(N/2)-1] = 1; u[int(N/2)+1] = -1
    epsG = 2e-19
    epsF = 0
    epsJ = 0
elif id==3:
    u = np.random.randn(N)
    u = np.zeros(N); u[int(N/2)] = 1
    u = gaussian_filter1d(u, 1.5)
    u[0]=u[-1]=0
    epsG = 1e-20
    epsF = 2e-18
    epsJ = 1e-18
elif id==4:
    v = np.abs(xx)
    v-=np.mean(v)
    u = v
u/=np.sum(np.abs(u))
plt.plot(xx/N, u.real)
plt.plot(xx/N, u.imag)
plt.show()

###
xx = np.linspace(-N/2,N/2,N*32,endpoint=False)
u_hat = recon.A(xx, N).dot(u)
plt.plot(xx, np.abs(u_hat))
plt.xlabel(r'$\xi$', fontsize=fontsize+2)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tight_layout(pad=0.5)
plt.show()
###
np.random.seed(3)
ngrid = 12*N
x = np.linspace(-N/2,N/2,ngrid)
X,Y=np.meshgrid(x,x)

bar = IncrementalBar('Pt', max = ngrid**2)
cf=np.zeros_like(X.reshape(-1))
g=np.zeros_like(X.reshape(-1))
f=np.zeros_like(X.reshape(-1))
normu2 = 0.5*np.linalg.norm(u)**2

for e,(xi1,xi2) in enumerate(zip(X.reshape(-1),Y.reshape(-1))):
    xi = np.array([xi1,xi2])
    loss, gradxi, x = recon.pseudoinv(xi, u, returngrad=False)
    cf[e] = loss

    ###
    matA = recon.A(xi, N)
    u_hat = matA.dot(u)
    matAt = np.conjugate(matA.T)
    f[e]=0.5*np.linalg.norm(u_hat)**2
    g[e] = loss+f[e]-normu2

    bar.next()
bar.finish()
cf = cf.reshape(ngrid,ngrid)
g = g.reshape(ngrid,ngrid)
f = f.reshape(ngrid,ngrid)



loc_min = utils.local_minima(g, eps=epsG)
pos = utils.mask_loc_min_to_scatter(loc_min, N)
fig, ax = plt.subplots(num=-1)
cb=plt.imshow(np.flip(g, axis=0), extent=[-N/2,N/2,-N/2,N/2])
cbar = plt.colorbar(cb)
cbar.ax.tick_params(labelsize=fontsize)
plt.scatter(pos[:,0],pos[:,1],color='red',marker='+',zorder=10)
plt.xticks(np.arange(N)-N/2)
plt.yticks(np.arange(N)-N/2)
ax.set_xticklabels([str(int(e-N/2)) if e%2==0 else '' for e,item in enumerate(ax.get_xticklabels())])
ax.set_yticklabels([str(int(e-N/2)) if e%2==0 else '' for e,item in enumerate(ax.get_yticklabels())])
plt.grid(True)
plt.xlabel(r'$\xi_1$', fontsize=fontsize+2)
plt.ylabel(r'$\xi_2$', fontsize=fontsize+2)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tight_layout(pad=0.5)
if save: plt.savefig("figures/inv_g"+str(id)+".pdf")

loc_min = utils.local_minima(-f, eps=epsF)
pos = utils.mask_loc_min_to_scatter(loc_min, N)
fig, ax = plt.subplots(num=-2)
cb=plt.imshow(np.flip(-f, axis=0), extent=[-N/2,N/2,-N/2,N/2])
cbar = plt.colorbar(cb)
cbar.ax.tick_params(labelsize=fontsize)
plt.scatter(pos[:,0],pos[:,1],color='red',marker='+',zorder=10)
plt.xticks(np.arange(N)-N/2)
plt.yticks(np.arange(N)-N/2)
ax.set_xticklabels([str(int(e-N/2)) if e%2==0 else '' for e,item in enumerate(ax.get_xticklabels())])
ax.set_yticklabels([str(int(e-N/2)) if e%2==0 else '' for e,item in enumerate(ax.get_yticklabels())])
plt.grid(True)
plt.xlabel(r'$\xi_1$', fontsize=fontsize+2)
plt.ylabel(r'$\xi_2$', fontsize=fontsize+2)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tight_layout(pad=0.5)
if save: plt.savefig("figures/inv_f"+str(id)+".pdf")

loc_min = utils.local_minima(cf, eps=epsJ)
pos = utils.mask_loc_min_to_scatter(loc_min, N)
fig, ax = plt.subplots(num=1)
cb=plt.imshow(np.flip(cf,axis=0), extent=[-N/2,N/2,-N/2,N/2])
cbar = plt.colorbar(cb)
cbar.ax.tick_params(labelsize=fontsize)
plt.scatter(pos[:,0],pos[:,1],color='red',marker='+',zorder=10)
plt.xticks(np.arange(N)-N/2)
plt.yticks(np.arange(N)-N/2)
ax.set_xticklabels([str(int(e-N/2)) if e%2==0 else '' for e,item in enumerate(ax.get_xticklabels())])
ax.set_yticklabels([str(int(e-N/2)) if e%2==0 else '' for e,item in enumerate(ax.get_yticklabels())])
plt.grid(True)
plt.xlabel(r'$\xi_1$', fontsize=fontsize+2)
plt.ylabel(r'$\xi_2$', fontsize=fontsize+2)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tight_layout(pad=0.5)
if save: plt.savefig("figures/inv_J"+str(id)+".pdf")

plt.show()
