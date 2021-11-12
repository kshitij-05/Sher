subroutine aotomo(nbasis,twoe,c,moint)
integer                 :: mu,nu,lam,sig,i,j,k,l,x,y,z,w,v,u
integer, intent(in)     :: nbasis
double precision,  intent(in)     :: twoe(nbasis,nbasis,nbasis,nbasis)  
double precision,  intent(in)     :: c(nbasis,nbasis)
double precision,  intent(out)        :: moint(nbasis,nbasis,nbasis,nbasis)
double precision        :: temp(nbasis,nbasis,nbasis,nbasis)
double precision        :: temp2(nbasis,nbasis,nbasis,nbasis)
double precision        :: temp3(nbasis,nbasis,nbasis,nbasis)
include "omp_lib.h"

!$OMP PARALLEL DO private(mu,nu,lam,sig,i,j,k,l,x,y,z,w,v,u) shared(nbasis,twoe,c,moint,temp,temp2,temp3) NUM_THREADS(4)
Do mu=1,nbasis
    Do i=1,nbasis
        Do x=1,nbasis
            Do y=1,nbasis
                Do z=1,nbasis
                    temp(z,y,x,mu) = temp(z,y,x,mu)+ c(mu,i)*twoe(z,y,x,i)
                end do
            end do
        end do
    end do
    Do nu=1,nbasis
        Do j=1,nbasis
            Do w=1,nbasis
                Do v=1,nbasis
                    temp2(v,w,nu,mu) = temp2(v,w,nu,mu) +c(nu,j)*temp(v,w,j,mu)
                end do
            end do
        end do
        Do lam=1,nbasis
            Do k=1,nbasis
                Do u=1,nbasis
                    temp3(u,lam,nu,mu) = temp3(u,lam,nu,mu)+c(lam,k)*temp2(u,k,nu,mu)
                end do
            end do
            Do sig=1,nbasis
                Do l=1,nbasis
                    moint(sig,lam,nu,mu) = moint(sig,lam,nu,mu)+c(sig,l)*temp3(l,lam,nu,mu)
                end do
            end do
        end do
    end do
end do
!$OMP END PARALLEL DO

end subroutine

subroutine newaotomo(nbasis,twoe,c,moint)
integer                 :: mu,nu,lam,sig,p,q,r,s
integer, intent(in)     :: nbasis
double precision,  intent(in)     :: twoe(nbasis,nbasis,nbasis,nbasis)  
double precision,  intent(in)     :: c(nbasis,nbasis)
double precision        :: moint(nbasis,nbasis,nbasis,nbasis)
!f2py intent(in,out)::moint
Do mu=1,nbasis
    Do p=1,nbasis
        Do nu=1,nbasis
            Do q=1,nbasis
                Do lam=1,nbasis
                    Do r=1,nbasis
                        Do sig=1,nbasis
                            Do s=1,nbasis
                                moint(s,r,q,p) = moint(s,r,q,p)+ c(p,mu)*c(q,nu)*c(r,lam)*c(s,sig)*twoe(sig,lam,nu,mu)
                            end do
                        end do
                    end do
                end do
            end do
        end do
    end do
end do
end subroutine





subroutine NUM_MULT(m,n,k)
integer                 :: m, n, k
intent(in)     :: m,n
intent(out)    :: k

k=m*n
end subroutine