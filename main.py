import numpy as np
import matplotlib.pyplot as plt

"""
main logic of code taken from https://www.youtube.com/watch?v=JFWqCQHg-Hs
"""

def main():
    Nx = 400
    Ny = 100
    Tau = 0.53
    Nt = 3000

    Nl = 9
    cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
    cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
    weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36])

    F = np.ones((Ny, Nx, Nl)) + 0.1 * np.random.randn(Ny, Nx, Nl)
    F[:,:,3] = 2.3

    X, Y = np.meshgrid(range(Nx), range(Ny))
    cyclinder = (X - Nx/2)**2 + (Y - Ny/2)**2 < 13**2
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
        ux  = np.clip(np.sum(F*cxs, 2) / rho, -0.4, 0.4)
        uy  = np.clip(np.sum(F*cys, 2) / rho, -0.4, 0.4)

        bnry = F[cyclinder, :]
        bnry = bnry[:, [0,5,6,7,8,1,2,3,4]]
        F[cyclinder, :] = bnry
        ux[cyclinder] = 0
        uy[cyclinder] = 0

        ux[engine] = 0.25

        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(Nl), cxs, cys, weights):
            Feq[:,:,i] = rho * w * (
                1 + 3*(cx*ux+cy*uy) + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2
            )

        F += -1/Tau * (F-Feq)

        if it%10 == 0:
            print(it)
            v = np.clip(np.sqrt(ux**2+uy**2), 0, 0.4)
            print(v.mean(), v.max(), v.std(), v.min())
            plt.imshow(v)
            plt.pause(0.01)
            plt.cla()



if __name__ == '__main__':
    main()