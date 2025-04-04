# SWGOLO
We use a parametrization of muon and electron+gamma fluxes as a function of radius R for energetic air showers, and a simplified description of detector units in terms of efficiency and acceptance, to model the measurement  of gamma rays by the SWGO array and to optimize the utility function of the experiment as a function of detector positions

# HOW TO USE
A description on how to run the code is reported in the `swgolo_xx.cpp` executable and also here for convenience

```
///////////////////////////////////////////////////////////////////////////////////////////////
//  Instructions to build and run project:                                                   //
//  --------------------------------------                                                   //
//  0. Create a directory for the project, e.g. /user/pippo/swgo/                            //
//  1. Copy this file to that directory                                                      //
//  2. Create subdirectories:                                                                //
//     - /user/pippo/swgo/Layouts     --> will contain plots of evolving config during run   //
//     - /user/pippo/swgo/Model       --> contains two txt files with model parameters       //
//     - /user/pippo/swgo/Dets        --> contains txt files with detector configurations    //
//     - /user/pippo/swgo/Outputs     --> contains dumps from runs of the program            //
//     - /user/pippo/swgo/Root        --> contains root files in output                      //
//  3. Copy model files into ./Model                                                         //
//     - Fit_Photon_10_pars.txt       --> parameters of photon model                         //
//     - Fit_Proton_2_pars.txt        --> parameters of proton model                         //
//  4. Copy detector layout files to ./Dets directory (optional)                             //
//  5. Modify this code to include the correct path to the main directory:                   //
//     Search and replace the string "/lustre/cmswork/dorigo/swgo/MT/"                       //
//                   with the string "/user/pippo/swgo/" (for example)                       //
//                   or change the variable GlobalPath                                       //
//  6. Decide which preprocessing directive to turn on: INROOT / STANDALONE (or UBUNTU),     //
//     and others - see "PREPROCESSING DIRECTIVES" below.                                    //
//                                                                                           //
//  IF running the code from linux (in mode called "STANDALONE" or "UBUNTU"):                //
//    7. Compile with:                                                                       //
//      > g++ -g swgolo148.cpp `root-config --libs --cflags` -O3 -o swgolo148                //
//    8. Run it by specifying all parameters you wish to test. E.g.:                         //
//      > nohup ./swgolo148 -nev 2000 -nba 2000 -nde 100 -nep 500 -sha 3 -nth 1  &           //
//      NB: parameters can also be set directly in the code (around line 10000) at the       //
//      start of the main() routine. This may be more practical for specific use cases.      //
//      Alternatively, if using taskset in multithreading, e.g. with 12 CPUs:                //
//      > nohup taskset -c 0-5, 16-21 ./swgolo147 -nth 12 &                                  //
//  ELSE IF running the code from root:                                                      //
//  7. Compile with:                                                                         //
//     > .L swgolo148.C+                                                                     //
//  8. Run it with:                                                                          //
//     > swgolo(2000,100,500);                                                               //
//     Specific options can be more practically hardwired in the code after line 10000,      //
//     below the definition  of the swgolo() function.                                       //
//  NNBB the code works with root v5.34.32 (latest version that has a windows release).      //
//  To run on root v6, there may be some fixed needed.                                       //
//                                                                                           //
//     IMPORTANT NOTES - caveat emptor                                                       //
//     -------------------------------                                                       //
//     1) the code is slow - the above parameters will produce one iteration                 //
//     in several minutes on a single CPU. with -nth you can speed this up almost            //
//     by the number of CPUs your system can deploy.                                         //
//     If running with RUNBENCHMARKS preprocessing directive set on, or if running with      //
//     one of the shapes 101-114, a single iteration (e.g. with 2000 simulated showers)      //
//     may take several hours on a single CPU. This is due to the CPU time quadratic         //
//     dependence on the number of detector units, and the fact that those shapes involve    //
//     values of Ndet in the few thousands range.                                            // 
//     2) Long comments in the code are sometimes outdated, wrong, or confusing.             //
//        Please do not make any reliance on long commented text that was written            //
//        during development and then never corrected / removed.                             //
//     3) Short comments are usually more reliable                                           // 
//     4) The code performs different tasks depending on the value of a few static           //
//        parameters (e.g., scanU, SameShowers, CheckUtility, SampleT, etcetera)             //
//        some of which can be changed during runtime. However, not all options are          //
//        guaranteed to work, and several are in conflict with others. Some checks are       //
//        performed at the start of running, but the user must be aware of what she is       //
//        doing when using parameter choices different from the default ones provided        //
//        in the code.                                                                       //
//     5) Some of the most tedious calculations of derivatives are obtained with the use     //
//        of Mathematica. See comments pointing to relevant notebooks.                       //
//                                                                                           //
//  Long-standing extensions to be worked at:                                                //
//  --------------------------------                                                         //
//  - implement averagetime_sigplusbgr in likelihood calculation (ignored so far)            //
//  - Treat separately electrons/positrons and gammas on the ground                          // 
//  - Implement efficiency for each particle type as integrated measure (indep. of Epart)    // 
//  - Implement resolution function per tank in Npart detection                              //
//  - Implement distribution of particle energies and E dependent efficiency                 //
//  - Implement more realistic model of fluxes versus polar angle of primary                 //
//  - Create valid model of secondaries arrival time (for now using flat arrival profile)    //
//  - Compute P(e->mu) as a function of theta and of position of other tanks and include it  //
//    in model and reconstruction                                                            //
///////////////////////////////////////////////////////////////////////////////////////////////
```
