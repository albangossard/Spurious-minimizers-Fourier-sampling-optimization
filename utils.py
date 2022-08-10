import numpy as np

def gen(P, N):
    # Signal generator
    f = np.zeros((P, N))
    for p in range(P):
        st = np.random.uniform(1, N-2)
        en = np.random.uniform(2, N-1)
        if en < st:
            st, en = en, st
        elif en == st:
            if en == N-1:
                st -= 1
            else:
                en += 1
        f[p, int(np.ceil(st)):int(en)] = 1
        if st != np.ceil(st):
            f[p, int(np.ceil(st))-1] = np.ceil(st)-st
        if en != int(en):
            f[p, int(en)] = en - int(en)
        f[p] /= np.linalg.norm(f[p])
    return f


def local_minima(array2d, eps=1e-5):
    return ((array2d <= -eps + np.roll(array2d,  (1,0), (0,1))) &
            (array2d <= -eps + np.roll(array2d, (-1,0), (0,1))) &
            (array2d <= -eps + np.roll(array2d,  (0,1), (0,1))) &
            (array2d <= -eps + np.roll(array2d, (0,-1), (0,1))) &
            (array2d <= -eps + np.roll(array2d,  (1,1), (0,1))) &
            (array2d <= -eps + np.roll(array2d,(-1,-1), (0,1))) &
            (array2d <= -eps + np.roll(array2d, (-1,1), (0,1))) &
            (array2d <= -eps + np.roll(array2d, (1,-1), (0,1)))
           )
def mask_loc_min_to_scatter(mask,N):
    ind = np.argwhere(mask==1)
    I=ind.reshape(-1)
    S = mask.shape[0]/N
    pos = I/S-N/2
    pos = pos.reshape(ind.shape)
    d=np.linalg.norm(pos.reshape(-1,1,2)-pos.reshape(1,-1,2), axis=2)
    l=[]
    marked = np.zeros(pos.shape[0])
    for i in range(pos.shape[0]):
        if marked[i]==0:
            bar = pos[i]
            cpt = 1
            for j in range(pos.shape[0]):
                if i!=j and d[i,j]<0.5 and marked[j]==0:
                    bar+=pos[j]
                    cpt+=1
                    marked[j] = 1
            marked[i] = 1
            l.append(bar/cpt)
    return np.array(l)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class BFGS():
    def __init__(self,nb_stock_max=8,verbose=1, test_curv=1e-14):
        self.nb_stock_max=nb_stock_max
        self.verbose=verbose
        self.stock=[]
        self.last_it=[]
        self.test_curv=test_curv
    def push(self, x, g):
        # x: current point
        # g: current gradient
        # returns the direction of descent
        self.shape = x.shape
        x=x.reshape(-1)
        g=g.reshape(-1)
        self.g=np.copy(g)
        if self.last_it:
            x_old,g_old=self.last_it
            curv=np.dot(x-x_old,g-g_old)
            if curv >self.test_curv:
                self.stock.append((np.copy(x-x_old),np.copy(g-g_old),1./curv))
            else:
                if self.verbose >0: print((bcolors.WARNING+'STOCK is emptied, curv={:1.3e}'+bcolors.ENDC).format(curv))
                self.stock=[]
            if len(self.stock) > self.nb_stock_max:
                self.stock.pop(0)
        self.last_it=(np.copy(x),np.copy(g))
    def get(self):
        return self.__find_direction(self.g).reshape(self.shape)

    def __find_direction(self, grad):
        if len(self.stock)==0:
            return -grad
        else:
            r=np.copy(grad)
            l_alpha=[]
            for (dx,dg,rho) in reversed(self.stock):
                alpha=rho*np.dot(dx,r)
                r=r-alpha*dg
                l_alpha.append(alpha)
            l_alpha=list(reversed(l_alpha))
            (dx,dg,rho)=self.stock[-1]
            gamma=np.dot(dx,dg)/np.dot(dg,dg)
            r=gamma*r
            for ((dx,dg,rho),alpha) in zip(self.stock,l_alpha):
                beta=rho*np.dot(dg,r)
                r=r+(alpha-beta)*dx
            return -r

class post_process():
    def __init__(self,save_cost=False,save_x=False,save_grad=False,save_d=False,save_step=False,save_info=False):
        self.to_save=[True,save_cost,save_x,save_grad,save_d,save_step,save_info]
        self.what=['niter','cost','x','grad','d','step','info']
        self.copy_function=[False,False,True,True,True,False,True]
        self.save={}
        for a,b in zip(self.what,self.to_save):
            if b:
                self.save[a]=[]
    def apply(self,niter,cost,x,grad,d,step,info):
        self.print(niter,cost,x,grad,d,step,info)
        self.do_save([niter,cost,x,grad,d,step,info])
        self.dummy(niter,cost,x,grad,d,step,info)
    def do_save(self,data):
        for a,b,c,d in zip(self.what,self.to_save,data,self.copy_function):
            if b:
                if d:
                    self.save[a].append(c.copy())
                else:
                    self.save[a].append(c)

    def print(self,niter,cost,x,grad,d,step,info):
        s='Iteration {:d}: cost {:1.3e} norm of x {:1.3e} norm of grad {:1.3e} step {:1.3e}'
        b=s.format(niter,cost,np.linalg.norm(x),np.linalg.norm(grad),step)
        print(b)


def Wolfe(x0,oracle,Niter=100,modify=None,e1=1.e-4,e2=0.9,initial_step=1.,post_process=None,tol_x=1.e-8,tol_g=1.e-12):
    if modify is None:
        modify=BFGS(0)
    if post_process is None:
        post_process=post_process()
    step=initial_step
    x=np.copy(x0)
    cost,grad,info=oracle(x)
    niter=0
    d=modify.apply(x,grad)
    post_process.apply(niter,cost,x,grad,d,step,info)
    Stop=False
    while not Stop:
        sm=0
        sp=np.infty
        Wolfe=True
        curv=np.dot(grad,d)
        niter_Wolfe=0
        if curv > 0:
            print('WARNING: ascent direction ({:1.3e}) at iteration {}'.format(curv,niter))
        while Wolfe:
            niter_Wolfe+=1
            x_tmp=x+step*d
            cost_tmp,grad_tmp,info_tmp=oracle(x_tmp)
            if sm > (1.-1.e-4)*sp or niter_Wolfe > 100:
                Wolfe=False
                print('LINE SEARCH FAILED TO CONVERGE',curv,cost_tmp-cost)
            elif cost_tmp > cost+e1*curv:
                sp=step
                step=0.5*(sm+sp)
            elif np.dot(grad_tmp,d) < e2*curv:
                    sm=step
                    step=min(0.5*(sm+sp),2*step)
            else:
                Wolfe=False
        niter+=1
        post_process.apply(niter,cost_tmp,x_tmp,grad_tmp,None,step,info_tmp)
        if np.linalg.norm(x_tmp-x) < tol_x*np.linalg.norm(x):
            Stop=True
            print('EXITING ALGORITHM  for small increases of x {:1.3e} << {:1.3e}'.format(np.linalg.norm(x_tmp-x),np.linalg.norm(x)) )
        if np.linalg.norm(grad_tmp) < tol_g:
            Stop=True
            print('EXITING ALGORITHM  for small gradient {:1.3e}'.format(np.linalg.norm(grad_tmp)) )
        if niter > Niter-1:
            Stop=True
            print('EXITING ALGORITHM  for iterations {}'.format(niter) )
        if not Stop:
            cost,x,grad,info=(cost_tmp,x_tmp,grad_tmp,info_tmp)
            d=modify.apply(x,grad)
    return post_process
