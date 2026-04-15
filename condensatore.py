import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from numba import jit

@jit
def matriceM(hx, hy, Nx, Ny):
    l_max=(Ny-1)*Nx+(Nx-1) #va da zero a NxNy-1
    #l_max=Nx
    M_lm=np.zeros(l_max+2*(l_max-1)+2*(l_max-Nx))
    #print(len(M_lm))
    for l in range(l_max): #vengono aggiunti prima elementi della diagonale, poi sopra, sotto, +Nx, -Nx (l'ordine di questi ultimi 4 non è a due a due rilevante)
        for m in range(l_max):
            if(l==m):
                if(l==0 or l==(l_max-1)):
                    M_lm[m]=1
                else:
                    M_lm[m]=-2*(1/pow(hx,2)+1/pow(hy,2))
            elif(m==(l+1)):
                if(l==0 or l==(l_max-1) or m==0 or m==(l_max-1)):
                    M_lm[l_max-1+m]=0
                else:
                    M_lm[l_max-1+m]=1/pow(hx,2)
            elif(m==(l-1)):
                if(l==0 or l==(l_max-1) or m==0 or m==(l_max-1)):
                    M_lm[l_max-1+(l_max-1)+1+m]=0
                else:
                    M_lm[l_max-1+(l_max-1)+m+1]=1/pow(hx,2)
            elif(m==(l+Nx)):
                if(l==0 or l==(l_max-1) or m==0 or m==(l_max-1)):
                    M_lm[l_max-1+2*(l_max-1)+l+1]=0
                else:
                    M_lm[l_max-1+2*(l_max-1)+l+1]=1/pow(hy,2)
                    
            elif(m==(l-Nx)):
                if(l==0 or l==(l_max-1) or m==0 or m==(l_max-1)):
                    M_lm[l_max-1+2*(l_max-1)+(l_max-Nx)+1+m]=0
                else:
                    M_lm[l_max-1+2*(l_max-1)+(l_max-Nx)+1+m]=1/pow(hy,2)
    return M_lm


@jit
def matriceMtilde(M_lm, Nx, Ny):
    l_max=Nx*Ny-1
    M_tilde=np.zeros(l_max+2*(l_max-1)+2*(l_max-Nx))
    for i in range (len(M_tilde)):
        if(i>=l_max and i<l_max+(l_max-1)):
            M_tilde[i]=-M_lm[i]/M_lm[i-l_max]
        elif(i>=l_max+(l_max-1)and i<l_max+2*(l_max-1)):
            M_tilde[i]=-M_lm[i]/M_lm[i-l_max-(l_max-1)]
        elif(i>=l_max+2*(l_max-1) and i<l_max+2*(l_max-1)+(l_max-Nx)):
            M_tilde[i]=-M_lm[i]/M_lm[i-(l_max+2*(l_max-1))]
        elif(i>=l_max+2*(l_max-1)+(l_max-Nx)):
            M_tilde[i]=-M_lm[i]/M_lm[i-(l_max+2*(l_max-1)+(l_max-Nx))]
        #print(M_tilde[i])
    return M_tilde
            
@jit         
def Jacobi(pot,M_tilde, bound, Nx):
    l_max=Nx*Nx-1
    potagg=np.copy(pot)
    resto=1
    conta=0
    #print(l_max)
    #print(len(pot))
    
    #for i in range(10000):
    while resto>10e-10:
        for i in range(len(pot)):
            if(bound[i]==True):
                potagg[i]=pot[i]
            else:
                potagg[i]=0#M_tilde[i]*pot[i]#elementi sulla diagonale
                if(i!=(len(pot)-1)): #se non siamo arrivati all'ultimo elemento
                    potagg[i]+=M_tilde[l_max+i]*pot[i+1]
                if(i!=0):#primo elemento
                    potagg[i]+=M_tilde[2*l_max-1+i]*pot[i-1]
                if((i+Nx)<len(pot)):
                    potagg[i]+=M_tilde[3*l_max-2+i]*pot[i+Nx]
                if((i-Nx)>=0 and 4*l_max-2-Nx+i<len(M_tilde)):
                    potagg[i]+=M_tilde[4*l_max-2-Nx+i]*pot[i-Nx]
        resto=np.sum(np.sqrt((potagg-pot)**2))
        #print(resto)
        conta+=1
        pot=np.copy(potagg)
        if(conta>=10000):       #più di così il mio  computer non regge
            break
            
    return potagg


def main():
    lato=1 #lato cella di simulazione
    L=0.7 #lunghezza piastre
    delta=0.1 #larghezza piastre
    V_1=100#potenziale piastra 1
    V_2=-100#potenziale piastra 2
    V_0=0#potenziale bordo
    points=int(input("Scegliere numero di punti: \n0 per 100 punti \n1 per 200 punti\n"))
    if(points==0):
        Nx=100
        Ny=100
    else:
        Nx=200
        Ny=200
    
    print(f"Risoluzione per Jacobi {Nx}x{Ny} :)")

    ax=0
    ay=0
    bx=1
    by=1
    hx=(bx-ax)/(Nx-1)
    hy=(by-ay)/(Ny-1)
    M_lm=matriceM(hx,hy,Nx,Ny)
    M_tilde=matriceMtilde(M_lm, Nx, Ny)

    potential=np.zeros((Ny,Nx))
    bound=np.zeros((Nx,Ny), dtype=bool)

    for i in range(Nx):
        potential[0][i]=V_0
        bound[0][i]=True
        potential[i][0]=V_0 #bordi a potenziale zero
        bound[i][0]=True
        potential[Nx-1][i]=V_0
        bound[Nx-1][i]=True
        potential[i][Nx-1]=V_0
        bound[i][Nx-1]=True
    for j in range((points+1)*10):
        for k in range ((points+1)*15,(points+1)*85):
            potential[k][(points+1)*15+j]=V_1
            bound[k][(points+1)*15+j]=True
            potential[k][(points+1)*75+j]=V_2
            bound[k][(points+1)*75+j]=True
    
    potential2=np.ravel(potential)
    bound2=np.ravel(bound)
    #print(len(potential))
    #print(len(M_tilde))
    #f=np.zeros((Nx,Ny))


    #calculating potential in every point
    potential_upd=Jacobi(potential2,M_tilde, bound2, Nx)
    
    #plotting potential
    V2D = potential_upd.reshape(Ny, Nx)
    X, Y = np.meshgrid(np.linspace(0,1,Nx), np.linspace(0,1,Ny))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, V2D, cmap='viridis')
    ax.set_title(f"Potenziale (Nx={Nx})")
    
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.contourf(X,Y,V2D, levels=100)
    plt.tight_layout()
    plt.show()
    
    #calculating the Laplacian of V: sum of second derivatives 
    lap=np.zeros((Nx,Nx))
    for i in range(1,Nx-1):
            for j in range(1,Nx-1):
                #lap[i][j]+=(V2D[i+1][j]-2*V2D[i][j]+V2D[i-1][j])/pow(hx,2)
                #lap[i][j]+=(V2D[i][j+1]-2*V2D[i][j]+potential[i][j-1])/pow(hy,2)
                lap[i][j]+=(potential_upd[(i+1)*Nx+j]-2*potential_upd[i*Nx+j]+potential_upd[(i-1)*Nx+j])/(pow(hx,2))    #d2/dx2
                lap[i][j]+=(potential_upd[i*Nx+j+1]-2*potential_upd[i*Nx+j]+potential_upd[i*Nx+j-1]) /pow(hy,2)  #d2/dy2

    e0=8.85e-12
    rho=-e0*lap

    '''
    rho2=np.zeros((len(rho),len(rho[0])))
    for i in range(len(rho)):
        for j in range(len(rho[0])):
            if(abs(rho[i][j])>=0.5e-10):
                rho2[i][j]=rho[i][j]
            else:
                rho2[i][j]=None
    '''  

    #graph of charge distribution
    rho2D = rho.reshape(Ny, Nx)
    X, Y = np.meshgrid(np.linspace(0,1,Nx), np.linspace(0,1,Ny))
    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection="3d")
    ax2.set_title("Densità di carica ")
    ax2.plot_surface(X, Y, rho2D, cmap='plasma')
    


    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.contourf(X,Y,rho2D, levels=100)
    plt.tight_layout()
    plt.show()
    


    #charge as double integral of charge distribution
    #naif rectangles method implemented
    charge=0
    #rho=abs(rho)
    Nxh=int(Nx/2)
    for k in range(Nx):
        for p in range(Nxh):
            charge+=rho[k][p]*hx*hy 
    
    '''
    charge3=0
    for k in range(Nx):
        for p in range(Nxh):
            if(bound[k][p]==True):
                charge3+=rho[k][p]*hx*hy
    '''
    #trapezoidal rule implemented
    charge2=0
    sum=0
    charge2+=rho[0][0]+rho[0][Nxh-1]+rho[Nx-1][0]+rho[Nx-1][Nxh-1]
    for k in range(2,Nx):
        for p in range(2,Nxh):
            sum+=rho[k][p]  
    charge2+=4*sum
    sum=0
    sum2=0
    for l in range(2,Nx-1):
        sum+=rho[0][l]+rho[Nx-1][l]
        sum2+=rho[l][0]+rho[l][Nxh-1]
    charge2+=2*sum
    charge2+=2*sum2
    charge2=((hx*hy)/4*charge2)

    Nxh = int(Nx/2) 


    #capacitance over height 
    #C/h=q/(h*V) -> q/h=lambda -> C/h=rho_piastra*V
    C_h=(charge/(2*V_1))
    C_h2=(charge2/(2*V_1))



    print("Carica 1 su singola armatura (metodo dei rettangoli naif): "+str(charge) +" C")
    print("Carica 2 su singola armatura (metodo dei trapezi): "+str(charge2) +" C")
    print("La capacità per unità di lunghezza (carica 1) è: " + str(C_h) + " F/m")
    print("La capacità per unità di lunghezza (carica 2) è: " + str(C_h2) + " F/m\n")
    

if __name__=='__main__':
    main()