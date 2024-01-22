import numpy as np
import matplotlib.pyplot as plt

"""
main logic of code taken from https://www.youtube.com/watch?v=JFWqCQHg-Hs
"""

def calc_mvn(ux, r, y0, x, n):
    """ суммирует v^n в круге x=x, |y0-y|<r 
        - расход (масса в секунду) n=1
        - для силы (dp/dt = dm/dt*v) n=2
        - для мощности n=3 
    по формуле sum(2 pi r dr * v^n) """
    return 2*np.pi*(np.sum(np.arange(r) * ux[y0:y0+r, x]**n) 
                    + np.sum(np.arange(r) * ux[y0-r+1:y0+1, x]**n))

def main():
    Nx = 400
    Ny = 100
    Tau = 0.53
    Nt = 3000
    Vmax = 0.4

    Nl = 9
    cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
    cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
    weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36])

    F = np.ones((Ny, Nx, Nl)) + 0.1 * np.random.randn(Ny, Nx, Nl)
    # F[:,:,3] = 2.3

    X, Y = np.meshgrid(range(Nx), range(Ny))
    # (X - Nx/2)**2 + (Y - Ny/2)**2 < 13**2 # cylinder
    # np.fill((Ny, Nx), False) # no obstacle
    obstacle = ((abs((Y - Ny/2)**2 + 6*(X - Nx/2)) < (Ny/2)) 
                & (abs(Y - Ny/2) > Ny/16) & (X > Nx/6)) # parabola
    engine = (abs(X-Nx/4) < Nx/20) & (abs(Y - Ny/2) < Ny/8)


    plt.figure(figsize=(16,8), dpi=80)

    for it in range(Nt):
        # print(it)
        
        # absorbing
        F[:, -1, [6,7,8]] = F[:, -2, [6,7,8]]
        F[:,  0, [2,3,4]] = F[:,  1, [2,3,4]]

        for i, cx, cy in zip(range(Nl), cxs, cys):
            F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
            F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)

        rho = np.sum(F, 2)
        ux  = np.clip(np.sum(F*cxs, 2) / rho, -Vmax, Vmax)
        uy  = np.clip(np.sum(F*cys, 2) / rho, -Vmax, Vmax)

        bnry = F[obstacle, :]
        F[obstacle, :] = bnry[:, [0,5,6,7,8,1,2,3,4]]
        ux[obstacle] = 0
        uy[obstacle] = 0

        ux[engine] = 0.13

        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(Nl), cxs, cys, weights):
            dot = cx*ux+cy*uy
            Feq[:,:,i] = rho * w * (1 + 3*(dot) + 9*(dot)**2/2 - 3*(ux**2+uy**2)/2)

        F += -1/Tau * (F-Feq)

        if it%50 == 0:
            print(it)
            v = np.clip(np.sqrt(ux**2+uy**2), 0, Vmax)
            print('mean max std min')
            print(v.mean(), v.max(), v.std(), v.min())
            print(f"At engine: Q={calc_mvn(ux, Ny//8, Ny//2, Nx//4+Nx//20, 1)}\n"
                  f"F={calc_mvn(ux, Ny//8, Ny//2, Nx//4+Nx//20, 2)}")
            print(f"At out: Q={calc_mvn(ux, 8, Ny//2, 200, 1)}\n"
                  f"F={calc_mvn(ux, 8, Ny//2, 200, 2)}")
            plt.imshow(v)
            plt.pause(0.01)
            plt.cla()



if __name__ == '__main__':
    main()