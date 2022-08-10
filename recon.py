import torch
import numpy as np


def A(xi, N):
    """"
    Creates a non-uniform Fourier matrix
    
    Parameters
    ----------
    xi : np.array
        Position of the points, should have shape (M,)
    N : int
        Dimension of the signal

    Returns
    -------
    np.array
        The Fourier matrix of shape (M, N)
    """

    nn = (np.arange(N)-N/2)/(N/2)
    A = np.exp(-1j*np.pi*xi.reshape(-1,1)*nn.reshape(1,-1))
    return A/np.sqrt(N)


def At(xi, N):
    """"
    Creates the adjoint of a non-uniform Fourier matrix
    
    Parameters
    ----------
    xi : np.array
        Position of the points, should have shape (M,)
    N : int
        Dimension of the signal

    Returns
    -------
    np.array
        The Fourier matrix of shape (N, M)
    """

    return np.conjugate(A(xi,N).T)


def backprop(xi, u, returngrad=True):
    if len(u.shape)==1:
        N = u.shape[0]
        matA = A(xi,N)
        matAt = np.conjugate(matA.T)
        x = matAt.dot(matA.dot(u))
        grad = np.zeros_like(xi)
        if returngrad:
            r = x - u
            nn = (np.arange(N)-N/2)/(N/2)
            grad_u_hat = matA.dot(u*(-1j*np.pi*nn))
            grad_r_hat = matA.dot(r*(-1j*np.pi*nn))
            grad = np.real(np.conjugate(grad_r_hat)*matA.dot(u) + np.conjugate(matA.dot(r))*grad_u_hat)
        return 0.5*np.linalg.norm( x-u )**2, grad, x
    elif len(u.shape)==2:
        uu = u.T
        N = uu.shape[0]
        matA = A(xi,N)
        matAt = np.conjugate(matA.T)
        x = matAt.dot(matA.dot(uu))
        grad = np.zeros_like(xi)
        if returngrad:
            r = x-uu
            nn = (np.arange(N)-N/2)/(N/2)
            grad_u_hat = matA.dot(uu*(-1j*np.pi*nn.reshape(-1,1)))
            grad_r_hat = matA.dot(r*(-1j*np.pi*nn.reshape(-1,1)))
            grad = np.real(np.conjugate(grad_r_hat)*matA.dot(uu) + np.conjugate(matA.dot(r))*grad_u_hat)
            grad = grad.sum(axis=1)/grad.shape[1]
        x = x.T
        return 0.5*np.linalg.norm( x-u, axis=1 )**2, grad, x
    else:
        raise Exception("The input signal should have 1 or 2 dimensions")


def pseudoinv(xi, u, returngrad=True):
    N = u.shape[0]
    matA = A(xi,N)
    matAt = np.conjugate(matA.T)
    B = matA.dot(matAt)
    y = matA.dot(u)
    eps = 1e-10
    x = matAt.dot(np.linalg.solve(B + eps*np.eye(B.shape[0]), y))
    grad = np.zeros_like(xi)
    if returngrad:
        raise NotImplementedError
    return 0.5*np.linalg.norm( x-u )**2, grad, x


def tikhonov(xi, f, lamb, returngrad=True, device=torch.device('cpu'), dtype=torch.float64):
    # xi (K,) array representing the frequencies in [-N/2,N/2]
    # f (L,N) array of signals
    # lamb regularization parameter
    freal = f.real
    fimag = f.imag
    ftorch = torch.zeros(f.shape[0], 2, f.shape[1], dtype=dtype, device=device)
    ftorch[:,0] = torch.tensor(freal.copy(), dtype=dtype, device=device)
    ftorch[:,1] = torch.tensor(fimag.copy(), dtype=dtype, device=device)
    ftorch = ftorch.view(f.shape[0], -1)

    K = xi.shape[0]
    N = f.shape[1]
    xitorch = torch.tensor(xi, dtype=dtype, device=device, requires_grad=returngrad)

    grid = torch.arange(N, device=device)-N/2
    A = torch.zeros(2*K, 2*N, dtype=dtype, device=device)
    A[:K,:N] = torch.cos(-2*np.pi* xitorch.view(-1,1)*grid.view(1,-1)/N)/np.sqrt(N)
    A[:K,N:] = -torch.sin(-2*np.pi*xitorch.view(-1,1)*grid.view(1,-1)/N)/np.sqrt(N)
    A[K:,:N] = torch.sin(-2*np.pi* xitorch.view(-1,1)*grid.view(1,-1)/N)/np.sqrt(N)
    A[K:,N:] = torch.cos(-2*np.pi* xitorch.view(-1,1)*grid.view(1,-1)/N)/np.sqrt(N)

    At = torch.zeros(2*N, 2*K, dtype=dtype, device=device)
    At[:N,:K] = torch.cos(2*np.pi* xitorch.view(1,-1)*grid.view(-1,1)/N)/np.sqrt(N)
    At[:N,K:] = -torch.sin(2*np.pi*xitorch.view(1,-1)*grid.view(-1,1)/N)/np.sqrt(N)
    At[N:,:K] = torch.sin(2*np.pi* xitorch.view(1,-1)*grid.view(-1,1)/N)/np.sqrt(N)
    At[N:,K:] = torch.cos(2*np.pi* xitorch.view(1,-1)*grid.view(-1,1)/N)/np.sqrt(N)

    C = torch.mm(At, A)
    b = torch.mm(C, torch.transpose(ftorch, 0, 1))
    D = C+lamb*torch.eye(2*N, device=device)
    x, _ = torch.solve(b, D)
    x = torch.transpose(x, 0, 1)

    loss = 0.5*(x-ftorch).pow(2).sum()/f.shape[0]

    gradxi = None
    if returngrad:
        loss.backward()

        gradxi = xitorch.grad.clone().cpu().numpy()
        xitorch.grad.data.zero_()

    x = x.view(f.shape[0], 2, f.shape[1])

    recon = np.zeros_like(f, dtype=np.complex)
    recon.real = x[:,0].detach().cpu().numpy()
    recon.imag = x[:,1].detach().cpu().numpy()
    return loss.item(), gradxi, recon
