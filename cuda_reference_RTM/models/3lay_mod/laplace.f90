program lapfilt
  implicit none
  character(len=64)	:: infile='dir.image', outfile='dir.imalap'
  integer		:: nz, nx, iz, ix
  real			:: dz, dx
  real, allocatable	:: i(:,:), o(:,:)

  ! Number of samples, first and last sample to be considered 
  nz = 151
  nx = 151
  dz = 10.
  dx = 10.

  allocate(i(nz,nx), o(nz,nx))
  i = 0.; o = 0.

  ! Read input data 
  open(unit=10, file=infile, form='unformatted', access='direct', recl=4*nz, status='unknown')
  do ix=1, nx 
  	read(10,rec=ix) (i(iz,ix), iz=1, nz)
  enddo
  close(10)

  ! Output data
  do ix=2, nx-1
	do iz=2, nz-1
		o(iz,ix) = (i(iz+1,ix)-2.*i(iz,ix)+i(iz-1,ix))/(dz*dz) + (i(iz,ix+1)-2.*i(iz,ix)+i(iz,ix-1))/(dx*dx)
	end do
  end do

  open(unit=20, file=outfile, form='unformatted', access='direct', recl=4*nz*nx, status='unknown')
  write(20,rec=1) o
  close(20)

  deallocate (i, o)
  return 
end program lapfilt 
