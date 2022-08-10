import numpy as np
import matplotlib.pyplot as plt
import argparse
import recon, utils


parser = argparse.ArgumentParser(description='Optimization of Fourier samples for 1D signals with the backprojection reconstruction')
parser.add_argument('--n', type=int, default=32, help='Number of values in the signal')
parser.add_argument('--m', type=int, default=16, help='Number of Fourier samples')
parser.add_argument('--p', type=int, default=1, help='Number of signals')
parser.add_argument('--s', type=float, default=5e-1, help='Step size')
parser.add_argument('--nit', type=int, default=200000, help='Number of iterations')
parser.add_argument('--freq', type=int, default=20000, help='Frequency to plot results')
parser.add_argument('--init', type=str, default='unif', help="Fourier samples initialization, can be 'unif', 'LF' or 'HF'")
parser.add_argument('--sto', action='store_true', help='Use stochastic updates')
parser.add_argument('--metric', action='store_true', help='Use variable metric')
parser.add_argument('--plot', action='store_true', help='Plot results during the optimization for debug purposes')
args = parser.parse_args()
N = int(args.n)
M = int(args.m)
P = int(args.p)
Niter = int(args.nit)
freq_aff = int(args.freq)
case = str(args.init)
sto = bool(args.sto)
metric = bool(args.metric)
plot = bool(args.plot)


# Initialization variables
np.random.seed(3)


xpname = case+'_P='+str(P)
if sto:
    xpname=xpname+'_sto'
if metric:
    xpname=xpname+'_metric'


if metric:
    f = utils.gen(5000, N)
    xiden = np.linspace(-N/2, N/2, 20*N)
    den = (np.abs(recon.A(xiden, N).dot(f.T))**2).sum(axis=1)/P
    den /= np.max(den)

# Image
if P==1 and not sto:
    f = np.zeros((1,N))
    bnd = int(N/4)
    f[0,bnd:-bnd] = 1
    f/=np.linalg.norm(f)
else:
    f = utils.gen(P, N)

if plot:
    plt.plot(f[0])
    plt.show()



if case=='unif':
    ## Uniform
    np.random.seed(2)
    xi=np.random.uniform(-np.pi,np.pi,M)
    step = 5e-1
elif case=='LF':
    ## LF (normal distribution)
    np.random.seed(1)
    xi = np.random.randn(M)*np.pi/4
    step = 5e-1
elif case=='HF':
    ## HF (uniform distribution)
    np.random.seed(1)
    xi = np.random.uniform(-3*np.pi/2,-np.pi/2,M)
    xi[xi<-np.pi] = xi[xi<-np.pi]+2*np.pi
    step = 5e-1
else:
    raise NotImplementedError

xi = xi.reshape(-1)*0.5*N/np.pi
if plot:
    plt.scatter(xi, np.zeros_like(xi))
    plt.show()

np.save("results/ADJ_1D_f_"+xpname+".npy", f)


CF=[]
PSNR=[]
list_normGrad=[]
traj = np.zeros((Niter+1, M))
traj[0] = xi.reshape(-1)
for nit in range(Niter):

    if sto:
        f = utils.gen(P, N)
    loss, gradxi, f_tilde = recon.backprop(xi, f)
    if metric:
        ind = np.ceil(((xi+N/2)%N)/(xiden[1]-xiden[0])).astype(np.int)%xiden.shape[0]
        gradxi /= den[ind]
    loss = np.mean(loss)
    psnr = np.mean(-10*np.log10( (np.abs(f-f_tilde)**2).sum(axis=1)/(np.abs(f)**2).sum(axis=1) ))
    normGrad = np.linalg.norm(gradxi)
    CF.append(loss)
    PSNR.append(psnr)
    list_normGrad.append(normGrad)
    xi = xi - step*gradxi
    traj[nit+1] = xi

    if (nit%freq_aff==freq_aff-1 or nit==Niter-1):
        print("It:%i -- CF:%1.2e -- PSNR:%1.3e -- s:%1.3e -- |\u2207|=%1.3e"%(nit,loss, psnr, step, normGrad ))
        if plot:
            plt.figure(1)
            plt.clf()
            plt.plot(PSNR)
            plt.grid(True)
            plt.tight_layout()
            plt.pause(0.01)

            plt.figure(2)
            plt.clf()
            plt.semilogy(CF-np.min(CF)+1e-16)
            plt.grid(True)
            plt.tight_layout()
            plt.pause(0.01)

            fig =plt.figure(3)
            plt.clf()
            ax = fig.add_subplot(1,1,1)
            for i in range(M):
                plt.plot(traj[:nit+1,i],np.arange(nit+1), color='blue', linewidth=2)
            plt.grid(True)
            ax.set_xlim(-N/2-1, N/2+1)
            plt.tight_layout()
            plt.pause(0.01)

            plt.figure(4)
            plt.clf()
            plt.semilogy(list_normGrad)
            plt.grid(True)
            plt.tight_layout()
            plt.pause(0.01)

        np.save("results/ADJ_1D_xi_"+xpname+".npy", xi)
        np.save("results/ADJ_1D_f_tilde_"+xpname+".npy", f_tilde)
        np.save("results/ADJ_1D_CF_"+xpname+".npy", CF)
        np.save("results/ADJ_1D_PSNR_"+xpname+".npy", PSNR)
        np.save("results/ADJ_1D_grad_"+xpname+".npy", list_normGrad)
        np.save("results/ADJ_1D_traj_"+xpname+".npy", traj)

if plot:
    plt.show()
