# STENCIL CODE

The source of the simples codes in CUDA and DPC++ are inside,respectively, the folders cuda_refenrence_stencil_computation, dpct_migrated_stencil_computation and dpct_migrated_stencil_computation_with_buffers.

## COMPILE

To these simple codes there is no need to compile a library.

### CUDA

There is a __Makefile__ inside the folder that can used to compile. To use the make command you need to be inside the folder that contains __Makefile__. An executble file is created inside the current folder.

### DPC++

There is a __Makefile__ inside the folder that can used to compile. To use the make command you need to be inside the folder that contains __Makefile__. An executble file is created inside the current folder.

## RUN

Is the same way to run CUDA and DPC++ codes. The images generated by the code will be inside the current directory.

```
$ ./stencil_code ./input.dat
```

# RTM CODE - CPU 

The folders of RTM codes in CUDA and DPC++ has four subdirectories: library (lib), velocity models (models), output and source (src).

## CUDA REFERENCE RTM CODE COMPILE

To compile and run the RTM code needs fist compile a library that has some auxilary functions. There is a __build.sh__ file in lib/src that can executed to compile these fuctions, to execute __build.sh__ type the command inside repository root folder:

```
cd cuda_reference_RTM/lib/src/
sh build.sh
```

If the compilation has finished without errors and produced an libsource.a in lib folder.

There is a __build.sh__ file in src that can executed or you can use make in the same path. To use __build.sh__ file or the make command you need to be inside the folder that contains __Makefile__. An executble file is created inside the current folder.

```
cd ../..
sh build.sh
```

## DPC++ REFERENCE RTM CODE COMPILE

First, start DPC++ environment:

```
source /opt/intel/oneapi/setvars.sh
```

To compile and run the RTM code needs fist compile a library that has some auxilary functions. There is a __build.sh__ file in lib/src that can executed to compile these fuctions, to execute __build.sh__ type the command inside repository root folder:

```
cd dpct_migrated_RTM/lib/src
make clean
make
```

If the compilation has finished without errors and produced an libsource.a in lib folder.

There is a __build.sh__ file in src that can executed or you can use make in the same path. To use __build.sh__ file or the make command you need to be inside the folder that contains __Makefile__. An executble file is created inside the current folder.

```
cd ../..
sh build.sh
```


## RUN

Is the same way to run CUDA and DPC++ codes. The images generated by the code will be inside the output directory.

```
./rtm_code ./models/<choose a model>/input.dat
```

> You may notice that some flags are missing in the compilation command in DPC ++ , only one flag match was found: __--fmad = false__. This flag was inserted in the main function of the source code in DPC++:  ___MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);__

# RTM CODE - GPU

# Pre-Requisites 
If you using os UBUNTU 
apt-get install libxt-dev xorg libx11-dev

Before run your code, go to lib/cwp folder, and: 
sh install.sh 

## COMPILE

To compile and run the RTM code needs fist compile. To compile go to src folder, and run make allclean, so run make. You need to be inside the folder that contains the file. If the compilation has finished without errors will are produced two executable files on build folder mod_main and rtm_main  in build folder. TO do that, execute this commands:

```
cd dpct_gpu_rtm_domain_division/lib/cwp/
sh install.sh
```

Then type "y" and "enter".

```
cd ../../src
make clean
make
cd..
```

## RUN
Before run rtm_main, you need to run modeling step: 
```
cd build/<choose_a_model>
../mod_main par=input.dat
```

After you can run normally the migration step:
```
../rtm_main par=input.dat
```# parallel-finite-difference-computation
