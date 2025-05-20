///////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                           //   
//                               S  W  G  O  L  O                                            //
//                                                                                           //   
//               Optimization of the footprint of SWGO detector array                        //
//               ----------------------------------------------------                        //
//                                                                                           //   
//  We use a parametrization of muon and electron+gamma fluxes as a                          //
//  function of radius R for energetic air showers, and a simplified description             //
//  of detector units in terms of efficiency and acceptance, to model the measurement        //
//  of gamma rays by the SWGO array and to optimize the utility function of the experiment   //
//  as a function of detector positions on the ground.                                       //
//                                                                                           //
//  This version of the code fits the position of showers (x0,y0) and their                  //
//  polar and azimuthal angles theta, phi and energy E through a likelihood maximization.    //
//  The fit is performed twice - once for the gamma and once for the proton hypothesis.      //
//  The two values of logL at maximum are used in the construction of a likelihood-ratio TS. // 
//  The distribution of this TS for the two hypotheses is the basis of the extraction        //
//  of the uncertainty on the signal fraction in the shower batches, with which a utility    //
//  value can be computed as f_s/sigma_f_s. Derivatives allow gradient descent to better     //
//  layouts of the array of detectors.                                                       //       
//                                                                 T. Dorigo, 2022 - 2024    //
//                                                                 with contributions from   //
//                                                                 L. Recabarren Vergara and //
//                                                                 M. Doro                   //
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

///////////////////////////////////////////////////////////////////////////////////////////////
// PREPROCESSING DIRECTIVES -------------------------------------------------------------------
// Please take care, no spaces are allowed before preprocessing directives (#xxx) at the start 
// of a line, as they will are interpreted as indentations/matching!
// --------------------------------------------------------------------------------------------

// Use one of the directives below alternatively, not both!
// --------------------------------------------------------
//#define PLOTS
#define FEWPLOTS // mandatory if scanU is true

// Ditto for the three below - pick INROOT for running under root, or one of the other two
// ---------------------------------------------------------------------------------------
//#define INROOT 
//
#define STANDALONE
//#define UBUNTU

// This is independent from the others, turn it on to get plots of E and angle res vs E and R
// ------------------------------------------------------------------------------------------
//#define PLOTRESOLUTIONS

// Turn this on if we have to run on one of the 14 benchmarks, with fewer events (dimensioning
// of Nunits is increased and the one of Nevents is decreased to avoid memory overflows) 
// -------------------------------------------------------------------------------------------
// #define RUNBENCHMARK
// Turn this on if we need to expand an array of Nunit macro-tanks of TankNumber tanks each
// into the full-fledged set of Nunit*TankNumber. This is because the dimension of the arrays
// need to match the number of required tanks.
// ------------------------------------------------------------------------------------------
// #define EXPANDARRAY
///////////////////////////////////////////////////////////////////////////////////////////////

#include "TH2.h"
#include "TH1.h"
#include "TF1.h"
#include "TLine.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TStyle.h"
#include "TColor.h"
#include "TCanvas.h" 
#include "TROOT.h"
#include "TFile.h"
#include "TMath.h"
#include "TRandom.h"
#include "TRandom3.h"
#include "Riostream.h"

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <utility>   // for std::pair
#include <algorithm> // for std::sort
#include <stack>

// The following is needed to use the timing feature, and for random seed setting
// ------------------------------------------------------------------------------
#include <ctime>

#if defined(STANDALONE) || defined(UBUNTU)
// For multithreading instructions
// -------------------------------
 #include <thread>
 #include <mutex>

// Define a mutex for synchronizing access to shared data
// Note, this is not presently used as the threading functions do not
// access shared variables.
// ------------------------------------------------------------------
std::mutex datamutex;

// Create multiple threads
// -----------------------
static std::vector<std::thread> threads;

// To improve multithreading
// -------------------------
 #include <pthread.h>
void SetAffinity(int thread_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(thread_id % std::thread::hardware_concurrency(), &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}
#endif // All the above is only valid for STANDALONE or UBUNTU use

using namespace std;

// UNITS: 
// ------
// E      in PeV
// length in meters
// time   in nanoseconds
// angles in radians

// Define a point structure to represent (x, y) coordinates
// --------------------------------------------------------
struct Point {
    double xx, yy;
};

// Forward declaration of gradient of reconstructed energy vs detector position
// ----------------------------------------------------------------------------
double dEk_dRik (int id, int is); 

// Forward declaration of voiding of region on ground
// --------------------------------------------------
void ForbiddenRegion (int type, int idstar, int mode);

// Forward declaration of routine resolving overlaps of tanks
// ----------------------------------------------------------
void ResolveOverlaps ();

// String handling different installations
// ---------------------------------------
#ifdef STANDALONE
static string GlobalPath = "/lustre/cmswork/dorigo/swgo/MT/";
#endif
#ifdef UBUNTU
static string GlobalPath = "/home/tommaso/Work/swgo/MT/";
#endif

// Constants and control settings
// ------------------------------ 
static const double ee             = 2.71828182845;
static const double largenumber    = 10000000000000.;
static const double epsilon        = 1./largenumber;
static const double epsilon2       = epsilon*epsilon;
static const double c0             = 0.29979;   // Speed of light in m/ns
static const double pi             = 3.1415926;
static const double twopi          = 2.*pi;
static const double halfpi         = 0.5*pi;
static const double sqrt2          = sqrt(2.);
static const double sqrt2pi        = sqrt(twopi);
static const double sqrtpi         = sqrt(pi);
static const double sqrt12         = sqrt(12.);
static const double log_10         = log(10.);
static const double log_01         = log(0.1);
static const double logdif         = log_10-log_01;
static const double thetamax       = pi*65./180.;
static const double beta1          = 0.8;       // Parameter of ADAM optimizer 
static const double beta2          = 0.99;      // Parameter of ADAM optimizer 
static const bool   debug          = false;     // If on, the code generates lots of printouts; use with caution 
static const bool   plotdistribs   = false;     // If on, plot densities per m^2 of shower profiles (the models from which Nmu, Ne are drawn)
static const bool   checkmodel     = false;     // If on, plot model distributions at the start
static const int    initBitmap     = 31;        // Binary map to initialize parameters of showers to their true values. 1=E, 2=P, 4=T, 8=Y0, 16=X0
static const bool   setXYto00      = false;     // If on, all showers hit the center of the array
static const bool   HexaShowers    = false;     // Used if fixShowerPos is true, to set the geometry of showers x,y centers
static const bool   CircleExposure = true;      // If fixShowerPos is false we can generate showers within a circle or a square
static const bool   checkUtility   = false;     // If on, the utility function is recomputed after a detector move, to check increase // under development
static int          idstar         = 35;        // Id of det for which gradient is computed if scanU option is on
static const bool   writeGeom      = true;      // If on, writes final detector positions to file
static const int    SampleT        = false;     // Whether sigmaLRT is sampled to get its variance
static const int    Nrep           = 10;        // Number of repetitions of T evaluation for determination of sigmaLRT (when SampleT is true)

// Default value of pass parameters
// --------------------------------
static double       GenGammaFrac   = 0.5;       // Fraction of photons in batches (NB in Nevents the fraction is always 0.5 to generate sound PDF distributions)
static bool         fixE           = false;     // If on, all showers are generated at Efix PeV
static double       Efix           = 0.2;       // 0.2 PeV
static bool         fixShowerPos   = false;     // If on, showers are always generated at the same locations - reduces stochasticity 
static bool         SameShowers    = false;     // If on, the same showers X0, Y0 are generated at each epoch, and theta,phi are set to zero; E is uniformly sampled
static bool         addSysts       = false;     // If on, we mess up a bit the distributions to mimic an imperfect knowledge of the models // not developed yet
static double       RelResCounts   = 0.05;      // Relative uncertainty in number of counts measured by tanks
static int Nevents                 = 3000;      // Number of showers to determine shape of TS at each step
static int Nbatch                  = 3000;      // Number of showers per batch
static int Nunits                  = 150;       // Number of detector units
static int Nactiveunits            = Nunits;    // Number of units that get moved around (see keep_fixed)
static int Nepochs                 = 2000;      // Number of SGD cycles performed
static double DetectorSpacing      = 100.;      // Spacing of detector units. Defined in call argument list
static double SpacingStep          = 100.;      // Additional parameter used to define some of the detector geometries
static double DisplFactor          = 2.;        // Used to control max displacement upon varying spacing and spacingstep
static double Rslack               = 2000.;     // This parameter controls how far away to generate showers around array area. Default changed since v123
static int shape                   = 3;         // 0 = hexagonal 1 = taxi 2 = spiral 3 = circular, and so on - this gets reassigned on start
static int CommonMode              = 3;         // Choice to vary all xy (0), R (1), or common center (2) of array during SGD
static double StartLR              = 1.0;       // Initial learning rate in SGD
static int Ngrid                   = 100;       // Number of xy points for initial assay of likelihood 
static int NEgrid                  = 10;        // Number of energy values of initial search for shower likelihood
static int Nsteps                  = 500;       // Number of steps in likelihood maximization
static double LRX                  = 1.;        // Learning rate on position in shower reconstruction likelihood
static double LRE                  = 0.05;      // Learning rate on energy in shower reconstruction likelihood
static double LRA                  = 0.05;      // Learning rate on angles in shower reconstruction likelihood
static double Eslope               = 0.;        // Slope of energy distribution for showers - NB neg -> larger importance to high E in U_GF
static const float coeff_GF        = 1.;        // These coefficients determine the relative importance of the parts of the utility,
static const float coeff_IR        = 50. ;      // and they are not used for determining the size of the move dx, dy; the size of the latter are determined
static const float coeff_PR        = 2000.;     // NNBB HACK wrt earlier versions // by the eta_xx coefficients below.
static const float coeff_TA        = 200.;      // Coefficient of area cost // was 500
static const float coeff_TL        = 100.;      // Coefficient of length cost
static const float coeff_PS        = 1000.;     // Coefficient of point source utility term
static double eta_GF               = 1.;        // Weight of gamma fraction in utility gradient
static double eta_IR               = 0.2;       // Weight of integrated resolution in utility gradient
static double eta_PR               = 0.0008;    // Weight of pointing resolution in utility gradient
static double eta_PS               = 0.000001;  // Weight of point source gradient
static double eta_TA               = 50000.;    // Weight of area utility gradient
static double eta_TL               = 10000.;    // Weight of area utility gradient
static int startEpoch              = 0;         // Starting epoch, for cases when continuing a run with readGeom true
static int Ntrigger                = 50;        // Minimum number of tanks recording >0 particles for an event to be counted
static int TankNumber              = 19;        // number of tanks in macroaggregates
static double TankRadius           = 1.91;      // 2.9 is tank D, 1.91 is tank A;      // Radius of element
static double minTankSpacing       = 0.6;       // Min distance between tanks
static double DefaultR2min         = 103.023;   // (holds for 19 2.9m tanks) This gets recomputed anyway by D2min() routine. In tanks macroaggregates, tanks of 1.91m are separated by 0.3m
static bool OrthoShowers           = false;     // If on, showers come down orthogonally to the ground (theta=0)
static bool SlantedShowers         = false;     // If on, showers come in at theta=pi/4
static bool usetrueXY              = false;     // If on, avoids fitting for x0,y0 of shower
static bool usetrueAngs            = false;     // If on, avoids fitting for theta, phi of shower
static bool usetrueE               = false;     // If on, avoids fitting for E of shower
static bool scanU                  = false;     // If on, U is scanned as a function of position of a detector unit
static bool readGeom               = false;     // If on, reads detector positions from file. Fix also startEpoch if true.
static bool PredefinedLayout       = true;      // read geometry from InputLayout.txt instead of searching for last file with same pars
static int PredefinedLayoutID      = 0;         // ID of input layout
static bool noSGDupdate            = false;     // Set to false if not changing detector layout
static bool initTrueVals           = true;      // if on, search for max L is performed starting with true values for parameters
static int Nthreads                = 32;        // Number of threads for cpu-intensive routines; this is default for C++ standalone code; set to 1 if running under root
static bool DynamicLR              = true;      // if on, LR of three components of utility are rescaled inversely to average logarithm of derivatives
static double maxDispl             = 400.;      // upper limit on step in detector xy position in an epoch; redefined in code
static bool UseAreaCost            = false;     // Either turn this one on or the one below
static bool UseLengthCost          = false;     // Either turn this one on or the one above (but they can also be both on)
static bool PeVSource              = false;     // If true, the utility is the one computed with the PeV source use case
static bool CrossingZenith         = false;     // For PeV sources, we distinguish ones at fixed latitude for polar angle generation
static double E_PS                 = 2.;        // PeV source energy
static double IntegrationWindow    = 128.;      // 128 nanoseconds integration window as per SWGO default. We assume we have two stages.
                                                // The logic is that we assume we trigger on a restricted area, and then use information on theta and phi
                                                // to collect hits that are consistent with the shower development.
static double Bgr_mu_per_m2        = 0.000001826*IntegrationWindow; // 1826/m2s within IntegrationWindow 
static double Bgr_e_per_m2         = 0.000000200*IntegrationWindow; // 200/m2s // this is a wild guess, but it is irrelevant due to much larger flux from any signal
                                                // From HAP-2024-004 (Conceicao, Pimenta et al.) cosmics at 5k altitude have nominal rate of 1826/m^2s
                                                // ---------------------------------------------------------------------------------------------------
static float SumProbRange          = 2.*sqrt(1.*Ntrigger);        // defining range where prob is evaluated
static bool dedrtrue               = false;     // Whether to remax the likelihood in calc of dedr
static double finalx_prevLR        = 0.;        // This gets read in if we reload conditions to continue a training
static double VoidRegion           = false;     // Enables the voiding of a part of the plane for detectors
static bool IncludeOnlyFR          = false;     // Flag that defines what is the allowed region, see ForbiddenRegion()
static bool KeepCentered           = true;      // Flag to decide if array has to be repositioned around origin of coordinates following GD updates
static float FRpar[3];                          // Parameters for definition of forbidden region
static double BackgroundConstant   = 1000000.;  // Multiplier for PeVSource background estimate (a flux normalization factor)

// Max dimensioning constants
// --------------------------
#if defined(RUNBENCHMARK) || defined(EXPANDARRAY) 
static const int    maxUnits       = 8371;      // Max number of detectors that can be deployed 
static const int    maxEvents      = 4000;      // Max events simulated for templates
#else
static const int    maxUnits       = 400;       // Max number of detectors that can be deployed 
static const int    maxEvents      = 10000;     // Max events simulated for templates (sum of Nevents and Nbatch)
#endif
static const int    maxTankNumber  = 61;        // Max number of tanks in an aggregate
static const int    minUnits       = 6;         // Lower limit on number of detectors
static const int    maxEpochs      = 10000;     // Max number of epochs of utility maximization
static const int    maxRbins       = 100;       // Max number of bins in R where to average utility (if CommonMode=1)
static const int    maxEinit       = 500;       // Max number of energy points for initial assay of shower likelihood
static const int    maxNtrigger    = 170;       // Max number of required units seeing a signal for triggering
static const int    maxNsteps      = 500.;      // Max steps in likelihood maximization. 300 works fine with adam optimizer (see below)
static const int    maxIter        = 20;        // Number of recorded trials in shower generation (for dU/dx calc.)
static const double maxinvrms      = 1000000.;  // Max value of inverse rms, see inv_rms_E function
static const bool   Debug_Rec      = false;     // generates same shower all over - Rename program when turning this true

// Global parameters required for dimensioning arrays, and defaults
// ----------------------------------------------------------------
static int NRbins                  = 10;        // Number of R bins in xy plane, to average derivatives of U (only relevant for CommonMode=1)
static int N_predef[14]            = { 6589, 
                                       6631, 
                                       6823, 
                                       6625, 
                                       6541, 
                                       6637, 
                                       6571, 
                                       4849, 
                                       8371, 
                                       3805, 
                                       5461, 
                                       5455, 
                                       4681, 
                                       3763 }; // number of units of 14 SWGO layouts

// Other parameters defining run
// -----------------------------
static double ArrayRspan[3]        = {0.,0.,0.};// Span in R of detector array. This gets defined based on the initial layout of x[], y[]
static double TotalRspan           = 0.;        // This also gets calculated in the code following definition of ArrayRspan vector
static double Xoffset              = 0.;        // Used to study behaviour of maximization and "drift to interesting region"
static double Yoffset              = 0.;        // Same as above for y: showers are "off center" with respect to the region where detectors are
static double XDoffset             = 0.;        // With these we put detector units offset instead of showers
static double YDoffset             = 0.;        // Same as above for y
static double Rmin                 = 2.;        // Minimum considered distance of shower center from detector (avoiding divergence of gradients) 
static int multiplicity            = 1;         // periodicity around circle
static int plotBitmap              = 5135;      // Binary map to sort out which graphs to fill and plot
static bool PlotThis[16];                       // Booleans corresponding to above bitmap
static double MaxUtility;                       // max value of plotted utility in U graph
static double TankArea;                         // 68.59 pi for 19 hexagonal shape macro units // 25.27pi for 7 1.9m radius "macro-tanks" aggregates in hexagonal pattern (works better than smaller units)
static const double deltapr        = 0.001;     // Regularization factor in position resolution calculation
static double deltapr2             = deltapr*deltapr; // see above
static double Wslope               = 0.;        // Slope with logE of weight of event for resolution contribution to utility.
static double maxdTdR              = 10.;       // ns variation of arrival time per m variation of Reff
static double Einit[maxEinit];                  // Initial energy values for scan of likelihood in E
static double U_IR_Num             = 0.;        // Numerator in integrated resolution contribution to utility
static double U_IR_Den             = 0.;        // Denominator in integrated resolution contribution to utility
static double U_PR_Num             = 0.;        // Numerator in pointing resolution contribution to utility
static double U_PR_Den             = 0.;        // Denominator in pointing resolution contribution to utility
static double U_GF                 = 0.;
static double U_IR                 = 0.; 
static double U_PR                 = 0.;
static double U_PS                 = 0.;
static double U_TA                 = 0.;
static double U_TL                 = 0.;
static double Rslack2;                          
static double dUA_dx;
static double dUA_dy;
static double dUL_dx;
static double dUL_dy;
static double dExpFac_dx;                       // Derivative of ExposureFactor for detector move dx
static double dExpFac_dy;                       // Same, for dy 
static double powbeta1[maxNsteps+1];
static double powbeta2[maxNsteps+1];
static double shift[10000];                     // Array of random Gaussian shifts (for histogramming)
static double Ng_active;                        // Number of gamma showers that are reconstructed in current batch
static double Np_active;                        // Number of proton showers that are reconstructed in current batch
static double N_active;                         // Total of above two numbers
static int    indfile;                          // This index contains the current id of the run for these parameters
static double EffectiveSpacing; 
static int    Nunits_npgt0;                     // Number of detectors with positive number of particles (used for DoF assessment)
static bool   StudyFluxRatio        = false;

// New random number generator
// ---------------------------
static TRandom3 * myRNG  = new TRandom3();
static TRandom3 * myRNG2 = new TRandom3();
static bool RndmSeed     = false;

// Shower parameters
// -----------------
static double PXeg1_p[3][3];
static double PXeg2_p[3][3];
static double PXeg3_p[3][3];
static double PXeg4_p[3][3];
static double PXmg1_p[3][3];
static double PXmg2_p[3][3];
static double PXmg3_p[3][3];
static double PXmg4_p[3][3];
static double PXep1_p[3][3];
static double PXep2_p[3][3];
static double PXep3_p[3][3];
static double PXep4_p[3][3];
static double PXmp1_p[3][3];
static double PXmp2_p[3][3];
static double PXmp3_p[3][3];
static double PXmp4_p[3][3];

// Parameters for lookup table of flux and derivatives
// ---------------------------------------------------
static double thisp0_mg[100][100];
static double thisp2_mg[100][100];
static double thisp0_eg[100][100];
static double thisp1_eg[100][100];
static double thisp2_eg[100][100];
static double thisp0_mp[100][100];
static double thisp2_mp[100][100];
static double thisp0_ep[100][100];
static double thisp1_ep[100][100];
static double thisp2_ep[100][100];
static double dthisp0de_mg[100][100];
static double dthisp2de_mg[100][100];
static double d2thisp0de2_mg[100][100];
static double d2thisp2de2_mg[100][100];
static double d3thisp0de3_mg[100][100];
static double d3thisp2de3_mg[100][100];
static double d2thisp0dth2_mg[100][100];
static double d2thisp2dth2_mg[100][100];
static double dthisp0de_eg[100][100];
static double dthisp1de_eg[100][100];
static double dthisp2de_eg[100][100];
static double d2thisp0de2_eg[100][100];
static double d2thisp1de2_eg[100][100];
static double d2thisp2de2_eg[100][100];
static double d3thisp0de3_eg[100][100];
static double d3thisp1de3_eg[100][100];
static double d3thisp2de3_eg[100][100];
static double d2thisp0dth2_eg[100][100];
static double d2thisp1dth2_eg[100][100];
static double d2thisp2dth2_eg[100][100];
static double dthisp0de_mp[100][100];
static double dthisp2de_mp[100][100];
static double dthisp0de_ep[100][100];
static double dthisp1de_ep[100][100];
static double dthisp2de_ep[100][100];
static double dthisp0dth_mg[100][100];
static double dthisp1dth_mg[100][100];
static double dthisp2dth_mg[100][100];
static double dthisp0dth_eg[100][100];
static double dthisp1dth_eg[100][100];
static double dthisp2dth_eg[100][100];
static double dthisp0dth_mp[100][100];
static double dthisp2dth_mp[100][100];
static double dthisp0dth_ep[100][100];
static double dthisp1dth_ep[100][100];
static double dthisp2dth_ep[100][100];
static double detA;                             // For computation of cubic interpolation of pars
static double Ai[4][4];                         // Same
static double A[4][4];                          // Matrix for solving cubics
static double B[4];                             // Initially contains values at four thetas, then converted in a,b,c,d of cubic
static double Y[4];                             // Values of cubic to interpolate fluxes
static double dBdY[4][4];                       // Derivatives of cubic pars vs Y
static double LLRmin; 
static double Utility;
static double UtilityErr;
static bool GammaHyp;
static double JS;                               // Jensen-Shannon divergence between the two densities of LLR

// Detector positions and parameters
// ---------------------------------
static double x[maxUnits];                      // Position of detectors in x
static double y[maxUnits];                      // Position of detectors in y 
static bool keep_fixed[maxUnits];               // This allows to fix the position of some of the units
static bool InMultiplet[maxUnits];              // Bool that tracks if an unit is part of a multiplet, or if it is not (if ForbiddenRegion has decoupled it)
static double xinit[maxUnits];                  // Initial position of detectors in x
static double yinit[maxUnits];                  // Initial position of detectors in y 
static double xprev[maxUnits];                  // Previous position in x 
static double yprev[maxUnits];                  // Previous position in y
static double ave_dUgf[maxUnits];               // average log of dUgf needed for dynamic rescaling of LR
static double ave_dUir[maxUnits];               // Average log of dUir needed for dynamic rescaling of LR
static double ave_dUpr[maxUnits];               // Average log of dUpr needed for dynamic rescaling of LR 
static double ave_dUtc[maxUnits];               // Average log of dUtc needed for dynamic rescaling of LR 
static double ave_dUa[maxUnits];                // Average log of dUa needed for dynamic rescaling of LR 
static double TrueX0[maxEvents];                // X of center of generated shower on the ground
static double TrueY0[maxEvents];                // Y of center of generated shower on the ground
static double TrueTheta[maxEvents];             // Polar angle of shower
static double TruePhi[maxEvents];               // Azimuthal angle of shower. phi=0 along x direction.
static double TrueE[maxEvents];                 // Energy of shower in PeV
static int Ntrials[maxEvents];                  // Used to generate showers only within Rslack from detectors
static int totNtrials;                          // Number of generated shower positions (used to determine effective area)
static double TryX0[maxEvents][maxIter];        // Recorded X0 position of showers used for acceptance calculation
static double TryY0[maxEvents][maxIter];        // Recorded Y0 position of showers
static bool Active[maxEvents];                  // Whether shower is considered (depends on number of tanks hit and P of likelihood fit)
static double PActive[maxEvents];               // Approximated probability that shower fires trigger, computed with exp values and Poisson approx
static double SumProbGe1[maxEvents];            // Needed for calculation of PActive above
static double F[maxNtrigger];                   // Lookup values of factorial, for sped up calcs
static double spiral_reduction = 0.999;         // Geometric factor for spiral layout, see definelayout function, option 2
static double step_increase    = 1.02;          // Geometric factor for spiral layout, see definelayout function, option 2
static double dU_dxi[maxUnits];                 // Derivative of the utility vs x detector
static double dU_dyi[maxUnits];                 // Derivative of the utility vs y detector
static double dU_dxiFR = 0.;                    // Gradient due to being near to forbidden region
static double dU_dyiFR = 0.;                    // Same as above, for y
static double sigma_time  = 10.;                // Time resolution assumed for detectors, 5 ns // was 1s before swgolo85
static double sigma2_time = pow(sigma_time,2);  // Squared time resolution
static double sigma_texp  = sigma_time;         // Time resolution of expected arrival of secondaries, for now equal to the one above
static double sigma2_texp = sigma2_time;        // Square of the above
static double pg[maxEvents];                    // Probability of test statistic for gamma hypothesis
static double pp[maxEvents];                    // Probability of test statistic for proton hypothesis
static double dIR_dEj[maxEvents];               // Variation of integrated resolution with energy
static int NumAvgSteps;
static int DenAvgSteps;
static double sumratio_dudxdy_GF = 0.;
static double sumratio_dudxdy_IR = 0.;
static double sumratio_dudxdy_PR = 0.;
static double sumratio_dudxdy_PS = 0.;
static double sumduy = 0.;

// Collected data from showers: Number of mus and es detected in each detector unit, average time of arrival in the tank 
// ---------------------------------------------------------------------------------------------------------------------
static int Nm[maxUnits][maxEvents];
static int Ne[maxUnits][maxEvents];
static float Tm[maxUnits][maxEvents];
static float Te[maxUnits][maxEvents];
// static float Stm[maxUnits][maxEvents];       // For the time being we do not use these terms (estimated uncertainties of time measurements)...
// static float Ste[maxUnits][maxEvents];       // ... as the calculation of likelihood derivatives would become significantly more complex. To be done.  
static double fluxB_mu;                         // These are counts from background noise
static double fluxB_e;

// Measured values of position and angle of shower
// -----------------------------------------------
static double x0meas[maxEvents][2];             // Measured X0 of shower. There are two values, one for each hypothesis (gamma/p)
static double y0meas[maxEvents][2];             // Measured Y0 of shower. 
static double thmeas[maxEvents][2];             // Measured theta of shower.
static double phmeas[maxEvents][2];             // Measured phi of shower.
static double e_meas[maxEvents][2];             // Measured energy of shower.

// Test statistic discriminating gamma from proton showers, for current batch
// --------------------------------------------------------------------------
static double logLRT[maxEvents];                // For templates construction
static bool   IsGamma[maxEvents];               // Whether event is generated as a true gamma or proton shower
static double sigmaLRT[maxEvents];              // RMS of Log-likelihood ratio, needed for derivative calculation
static double dsigma2_dx[maxUnits][maxEvents];  // Derivative of LLR uncertainty squared of shower ik over effective distance x from detector i
static double dsigma2_dy[maxUnits][maxEvents];  // Derivative of LLR uncertainty squared of shower ik over effective distance y from detector i
static double InvRmsE[maxEvents];               // This is filled by inv_rms_E()
static double TrueGammaFraction;                // Fraction of gammas in generated batch
static double MeasFg;                           // Measured fraction of photons in batch from Likelihood using pg, pp PDF values
static double MeasFgErr;                        // Uncertainty on fraction as derived from fit to TS distribution
static double inv_sigmafs;
static double inv_sigmafs2;
static double sigmafs2;
static double ExposureFactor;                   // This factor multiplies the utility as it scales with the area illuminated by generated showers
static double Emin                  = 0.1;      // In PeV. Lower boundary of energy in shower model
static double Emax                  = 10.;      // Upper boundary
static const double MinLearningRate = 0.5;      // Min value of LR for SGD
static const double MaxLearningRate = 2.0;      // Max value of LR for SGD
static const double logLApprox      = 0.1;      // Determines precision of shower reconstruction likelihood (decreasing it slows down calcs)
static int N_pos_derivative         = 0;        // Counter of screwups in inv_rms calculation
static float AverLastXIncr          = 0.;       // Check of convergence of LnL X determination
static float AverLastYIncr          = 0.;       // Check of convergence of LnL Y determination
static float AverLastEIncr          = 0.;       // Check of convergence of LnL E determination
static float AverLastTIncr          = 0.;       // Check of convergence of LnL T determination
static float AverLastPIncr          = 0.;       // Check of convergence of LnL P determination
static int   NAverLastIncr          = 0;  

// Static variables for point source utility and derivatives
// ---------------------------------------------------------
static bool   useN5s                = true;    // Switch for which U_PS term to use
static double PS_vardth             = 0.;
static double PS_vardph             = 0.;
static double PS_varde              = 0.;
static double PS_SidebandArea       = 0.;
static double PS_TotalSky           = 0.;
static double PS_EintervalWeight    = 0.;
static double PS_B                  = 0.;
static double PS_dBdx               = 0.;
static double PS_dBdy               = 0.;
static double PS_SumWeightsE        = 0.;
static double PS_SumWeightsE2       = 0.;
static double PS_sigmaBoverB        = 0.;       
static double PS_dsigmaBoverB_dx    = 0.;       // This and the one below are filled in the code during loop on id in threadfunction2, before calling N_3S
static double PS_dsigmaBoverB_dy    = 0.;       // NNBB - only call N_3S(mode=1,2) after these are correctly filled.
static double PS_dnum1dxj[maxUnits];
static double PS_dnum2dxj[maxUnits];
static double PS_dnum1dyj[maxUnits];
static double PS_dnum2dyj[maxUnits];
static double PS_sumdPAkdxj[maxUnits];
static double PS_sumdPAkdyj[maxUnits];
static double PS_sumdPAkdxjw[maxUnits];
static double PS_sumdPAkdyjw[maxUnits];
static double PS_dsigmaEi_dxj[maxUnits];
static double PS_dsigmaEi_dyj[maxUnits];

// Those below are used to check the effectiveness of grid initialization versus true initial values
// -------------------------------------------------------------------------------------------------
static int Start_true_trials    = 0;
static int Start_true_wins      = 0;
static bool CheckInitialization = false;
static double sumdup            = 0.;
static double sumduc            = 0.;
static float sumrat             = 0.;
static int nsum                 = 0;
static int warnings1            = 0.;
static int warnings2            = 0.;
static int warnings3            = 0.;
static int warnings4            = 0.;
static int warnings5            = 0.;
static int warnings6            = 0.;
static int warnings7            = 0.;
static ofstream outfile;

// Model of N3S, the number of events for a 3-sigma excess in UtilityPevSource
// ---------------------------------------------------------------------------
static double N3s_p00 = 2.78868;
static double N3s_p01 = 2.81895;
static double N3s_p02 = 0.820116;
static double N3s_p10 = -5.44998;
static double N3s_p11 = 0.0080324;
static double N3s_p12 = 9.41535;
static double N3s_p13 = 2.;
static double N3s_p20 = -6.83767;
static double N3s_p21 = 12.0905;
static double N3s_p22 = 1.02073;

// Static histograms (ones that are handled in called functions)
// -------------------------------------------------------------
static int nfilled = 0;
static TH1D *dEdR[10];
static double maxdxy      = 2000.;
static TProfile * DE         = new TProfile   ("DE",         "Energy resolution vs true energy", 10, 0., 10., 0., 1000.); 
static TProfile * DE0        = new TProfile   ("DE0",        "Energy resolution vs true energy", 10, 0., 10., 0., 1000.); 
static TProfile * DR         = new TProfile   ("DR",         "Log(pointing resolution) vs E", 10, 0., 10., -10., 10.); 
static TProfile * DR0        = new TProfile   ("DR0",        "Log(pointing resolution) vs E", 10, 0., 10., -10., 10.); 
static TH2D * ThGIvsThGP     = new TH2D       ("ThGIvsThGP", "Angle GI vs angle GP",    10, 0., pi, 10, 0., pi);
static TH2D * LrGIvsLrGP     = new TH2D       ("LrGIvsLrGP", "Size dUdx GI vs DUdx GP", 10, 0., pi, 10, 0., pi);
//static TH1D * numx = new TH1D ("numx", "", 100, -10., 20.);
//static TH1D * numy = new TH1D ("numy", "", 100, -10., 20.);
//static TH1D * denx = new TH1D ("denx", "", 100, -10., 20.);
//static TH1D * deny = new TH1D ("deny", "", 100, -10., 20.);
//static TH1D * dxrec = new TH1D ("dxrec", "", 100, -10., 10.);
//static TH1D * dyrec = new TH1D ("dyrec", "", 100, -10., 10.);

#ifdef PLOTRESOLUTIONS
    static TProfile * DE0vsE     = new TProfile   ("DE0vsE",     "Rel. energy resolution vs E/PeV", 20, 0., 10., 0., 1000.); 
    static TProfile * DE0vsR     = new TProfile   ("DE0vsR",     "Rel. energy resolution vs log(R/m)", 20, 4., 8., 0., 1000.); 
    static TProfile2D * DE0vsER  = new TProfile2D ("DE0vsER",    "Rel. energy resolution", 20, 4., 8., 20, 0., 10., 0., 1000.); 
    static TProfile * DR0vsE     = new TProfile   ("DR0vsE",     "Pointing resolution vs E/PeV", 20, 0., 10., 0., 1000.);
    static TProfile * DR0vsR     = new TProfile   ("DR0vsR",     "Pointing resolution vs log(R/m)", 20, 4., 8., 0., 1000.);
    static TProfile2D * DR0vsER  = new TProfile2D ("DR0vsER",    "Pointing resolution", 20, 4., 8., 20, 0., 10., 0., 1000.); 
    static TProfile2D * NvsER    = new TProfile2D ("NvsER",      "N entries per epoch in ER graph", 20, 4., 8., 20, 0., 10., 0., 100000);
#endif

// Matrices needed for resolution calculations
// -------------------------------------------
static double meandE[20][20];
static double meandE2[20][20];
static double denE[20][20];
static double meandR[20][20];
static double meandR2[20][20];
static double denR[20][20];
static double meandE_E[20];
static double meandE2_E[20];
static double denE_E[20];
static double meandR_E[20];
static double meandR2_E[20];
static double denR_E[20];
static double meandE_R[20];
static double meandE2_R[20];
static double denE_R[20];
static double meandR_R[20];
static double meandR2_R[20];
static double denR_R[20];
//#ifdef FEWPLOTS
static TH1D * DUGF        = new TH1D     ("DUGF", "dU/dx flux term",   75, -15., 10.);
static TH1D * DUIR        = new TH1D     ("DUIR", "dU/dx energy term", 75, -15., 10.);
static TH1D * DUPR        = new TH1D     ("DUPR", "dU/dx angle term",  75, -15., 10.);
static TH1D * DUTC        = new TH1D     ("DUTC", "dU/dx cost term",   75, -15., 10.);
static TProfile * SvsSP   = new TProfile ("SvsSP", "RMS of test statistic comparison", 100, 0., 10., 0., 20.);
static TH2D * SvsS        = new TH2D     ("SvsS",  "RMS of test statistic comparison", 100, 0., 10., 100, 0., 10.);
// static TH1D * TT          = new TH1D     ("TT",    "Theta of showers", 100, 0., pi*70./180);
static TH2D * NumStepsvsxy;
static TH2D * NumStepsvsxyN; 
//#endif


#ifdef PLOTS
// The ones below check the triggering probability vs distance from shower core
// ----------------------------------------------------------------------------
static TH2D *     PGvsD   = new TH2D     ("PGvsD",  "G probability of triggering vs D and E", 25, 0., 2500., 10, 0., 10.); // to do these plots update to 2500 rmax
static TH2D *     PPvsD   = new TH2D     ("PPvsD",  "P probability of triggering vs D and E", 25, 0., 2500., 10, 0., 10.);
static TH2D *     PAGvsD  = new TH2D     ("PAGvsD", "G probability of triggering vs D and E", 25, 0., 2500., 10, 0., 10.);
static TH2D *     NPAGvsD = new TH2D     ("NPAGvsD","G probability of triggering vs D and E", 25, 0., 2500., 10, 0., 10.);
static TH2D *     PAPvsD  = new TH2D     ("PAPvsD", "P probability of triggering vs D and E", 25, 0., 2500., 10, 0., 10.);
static TH2D *     NPAPvsD = new TH2D     ("NPAPvsD","P probability of triggering vs D and E", 25, 0., 2500., 10, 0., 10.);

//static TH2D * NumStepsvsxy;
//static TH2D * NumStepsvsxyN; 
//static TProfile * SvsSP = new TProfile  ("SvsSP",     "RMS of test statistic comparison", 100, 0., 10., 0., 20.);
//static TH2D * SvsS      = new TH2D      ("SvsS",      "RMS of test statistic comparison", 100, 0., 10., 100, 0., 10.);
static TH1D * DXP         = new TH1D      ("DXP",       "Difference between true and fit x for proton showers", 200, -maxdxy, maxdxy);
static TH1D * DYP         = new TH1D      ("DYP",       "Difference between true and fit y for proton showers", 200, -maxdxy, maxdxy);
static TH1D * DXG         = new TH1D      ("DXG",       "Difference between true and fit x for gamma showers",  200, -maxdxy, maxdxy);
static TH1D * DYG         = new TH1D      ("DYG",       "Difference between true and fit y for gamma showers",  200, -maxdxy, maxdxy);
static TH1D * DTHG        = new TH1D      ("DTHG",      "Difference between true and fit theta for gamma showers",  200, -halfpi, halfpi);
static TH1D * DPHG        = new TH1D      ("DPHG",      "Difference between true and fit phi for gamma showers",    200, -twopi, twopi);
static TH1D * DTHP        = new TH1D      ("DTHP",      "Difference between true and fit theta for proton showers", 200, -halfpi, halfpi);
static TH1D * DPHP        = new TH1D      ("DPHP",      "Difference between true and fit phi for proton showers",   200, -twopi, twopi);
static TH1D * DEG         = new TH1D      ("DEG",       "Difference between true and fit energy for gamma showers",   100, 0., 10.);
static TH1D * DEP         = new TH1D      ("DEP",       "Difference between true and fit energy for proton showers",  200, 0., 5.);
static TH2D * DTHPvsT     = new TH2D      ("DTHPvsT",   "Theta residual for proton showers vs theta", 50, -halfpi,halfpi,50, 0.,halfpi );
static TH2D * DTHGvsT     = new TH2D      ("DTHGvsT",   "Theta residual for gamma showers vs theta",  50, -halfpi,halfpi,50, 0.,halfpi );
// Histograms for calculation of likelihood performance vs R
// ---------------------------------------------------------
static const int NbinsResR = 20;
static const int NbinsResE = 10;
static TH2D * XRMSvsRg    = new TH2D     ("XRMSvsRg",   "RMS of X estimate vs distance from center", NbinsResR, 0., 1500., NbinsResE, 0.,10.);
static TH2D * ERMSvsRg    = new TH2D     ("ERMSvsRg",   "RMS of E estimate vs distance from center", NbinsResR, 0., 1500., NbinsResE, 0.,10.);
static TH2D * TRMSvsRg    = new TH2D     ("TRMSvsRg",   "RMS of T estimate vs distance from center", NbinsResR, 0., 1500., NbinsResE, 0.,10.);
static TH2D * PRMSvsRg    = new TH2D     ("PRMSvsRg",   "RMS of P estimate vs distance from center", NbinsResR, 0., 1500., NbinsResE, 0.,10.);
static TH2D * XRMSvsRp    = new TH2D     ("XRMSvsRp",   "RMS of X estimate vs distance from center", NbinsResR, 0., 1500., NbinsResE, 0.,10.);
static TH2D * ERMSvsRp    = new TH2D     ("ERMSvsRp",   "RMS of E estimate vs distance from center", NbinsResR, 0., 1500., NbinsResE, 0.,10.);
static TH2D * TRMSvsRp    = new TH2D     ("TRMSvsRp",   "RMS of T estimate vs distance from center", NbinsResR, 0., 1500., NbinsResE, 0.,10.);
static TH2D * PRMSvsRp    = new TH2D     ("PRMSvsRp",   "RMS of P estimate vs distance from center", NbinsResR, 0., 1500., NbinsResE, 0.,10.);
static TH1D * DXYg[NbinsResR*NbinsResE];
static TH1D * DE_g[NbinsResR*NbinsResE];
static TH1D * DT_g[NbinsResR*NbinsResE];
static TH1D * DP_g[NbinsResR*NbinsResE];
static TH1D * DXYp[NbinsResR*NbinsResE];
static TH1D * DE_p[NbinsResR*NbinsResE];
static TH1D * DT_p[NbinsResR*NbinsResE];
static TH1D * DP_p[NbinsResR*NbinsResE];
static TH1D * SigLRT      = new TH1D      ("SigLRT",    "", 100, 0., 100000.);
static TH2D * SigLvsDRg   = new TH2D      ("SigLvsDRg", "", 100, 0., 100000., 100, 0., 1000.);
static TH2D * SigLvsDRp   = new TH2D      ("SigLvsDRp", "", 100, 0., 100000., 100, 0., 1000.);
static TH2D * DL          = new TH2D      ("DL",        "", 100, 0.5, 100.5,100,-50000.,+50000.);
static TH2D * NmuvsSh     = new TH2D      ("NmuvsSh",   "", 200, 0., 2000., 50, 0., 50.);
static TH2D * NevsSh      = new TH2D      ("NevsSh",    "", 200, 0., 2000., 50, 0., 50.);
static TProfile * NumStepsg; 
static TProfile * NumStepsp;  
static TH2D * P0mg        = new TH2D      ("P0mg","", 100, 0., 100., 100, 0., 100.);
static TH2D * P2mg        = new TH2D      ("P2mg","", 100, 0., 100., 100, 0., 100.);
static TH2D * P0mp        = new TH2D      ("P0mp","", 100, 0., 100., 100, 0., 100.);
static TH2D * P2mp        = new TH2D      ("P2mp","", 100, 0., 100., 100, 0., 100.);
static TH2D * P0eg        = new TH2D      ("P0eg","", 100, 0., 100., 100, 0., 100.);
static TH2D * P1eg        = new TH2D      ("P1eg","", 100, 0., 100., 100, 0., 100.);
static TH2D * P2eg        = new TH2D      ("P2eg","", 100, 0., 100., 100, 0., 100.);
static TH2D * P0ep        = new TH2D      ("P0ep","", 100, 0., 100., 100, 0., 100.);
static TH2D * P1ep        = new TH2D      ("P1ep","", 100, 0., 100., 100, 0., 100.);
static TH2D * P2ep        = new TH2D      ("P2ep","", 100, 0., 100., 100, 0., 100.);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functions
// ---------

// Log factorial function
// ----------------------
long double LogFactorial (int n) {
    // Use Stirling's approximation
    if (n==0) return 0.;
    return log(sqrt(twopi*n)) + (double)n*(log(n)-1.); 
}  

// Factorial function
// Note, it breaks down for N>170. Better use logfactorial below
// -------------------------------------------------------------
long double Factorial (int n) {
  if (n==0)  return 1.;
  if (n>170) { 
    // cout << "     Warning - Factorial calculation breaks down for this n = " << n << endl;
    // return largenumber;
    return exp(LogFactorial(n));
  }
  return Factorial(n-1)*n;
}

// Function that computes phi from x,y
// -----------------------------------
double PhiFromXY (double thisx, double thisy) {
    double phi = 0.;
    if (thisx!=0.) { 
        phi = atan(thisy/thisx);
        if (thisx<0.) {
            phi += pi;
        } else {
            if (thisy<0.) {
                phi = 2.*pi+phi;
            }
        }
    } else {
        phi = pi/2.;
        if (thisy<0.) phi += pi;
    }
    return phi;
}

// Old Poisson function, with approximation for large numbers
// ----------------------------------------------------------
double OldPoisson (int k, double mu) {
    double p;
    if (mu<0.) {
        cout    << "Mu<0 in oldpoisson" << endl;
        outfile << "Mu<0 in oldpoisson" << endl;
        warnings5++;
        return 0.;
    } else if (mu==0.) {
        return 0.;
    } else if (mu<15. && k<maxNtrigger) {
            p = exp(-mu)*pow(mu,k)/F[k];
    } else {
        // Use Gaussian approximation
        // --------------------------
        double s = sqrt(mu);
        p = exp(-0.5*pow((k-mu)/s,2.))/(sqrt2pi*s);
    }
    return p;
}

// Poisson function, with approximation for large numbers
// ------------------------------------------------------
double MyPoisson (int k, double mu) {
    double p;
    if (mu<0.) {
        cout    << "Mu<0 in mypoisson" << endl;
        outfile << "Mu<0 in mypoisson" << endl;
        warnings5++;
        return 0.;
    } else if (mu==0.) {
        return 0.;
    } else if (mu<65. && k<maxNtrigger) { // for mu>=67 and k=170 it starts to give inf
        p = exp(-mu)*pow(mu,k)/F[k];
    } else {
        // Use Stirling approximation
        double logp = -mu + k*log(mu) - LogFactorial(k);
        p = exp(logp);
    }
    if (p<0. || p>1.) {
        cout    << "Sorry, p = " << p << " for k,mu = " << k << " " << mu << endl;
        outfile << "Sorry, p = " << p << " for k,mu = " << k << " " << mu << endl;
    }
    return p;
}

// Function that smears the expected counts 
// ----------------------------------------
float SmearN (float flux) {
    if (!addSysts) return flux; // just making sure
    if (flux==0.) return flux;
    // Use a different random number generator here, to avoid messing up the showers if SameShowers is on
    return myRNG2->Gaus(flux,RelResCounts*flux); // Assume that the counting resolution is a linear function of N
}

// Learning rate scheduler - this returns an oscillating, dampened function as a function of the epoch
// ---------------------------------------------------------------------------------------------------
double LR_Scheduler (int epoch) {
    double par[3] = {-0.03,0.3,0.2}; 
    double x = 100.*(epoch-startEpoch)/Nepochs + finalx_prevLR;
    // We include a value StartLR_Thisrun set to 1. for runs with no previous history,
    // and read in from file in case of continuing training
    // -------------------------------------------------------------------------------
    return exp(par[0]*x)*(par[1]+(1.-par[1])*pow(cos(par[2]*x),2));
}

// Derivation of the distance from the axis: we define the point on the axis of minimal distance
// to (x,y,0) as (a,b,c). Coordinates (x=a(t),y=b(t),z=c(t)) of points on the line fulfil the conditions
//    a = t * sin(theta) * cos(phi);
//    b = t * sin(theta) * sin(phi);
//    c = t * cos(theta). 
// [NB With this parametrization, positive t means negative z as phi is angle of positive propagation downward]
// We express the distance between the two points as
//    d^2 = (x-a)^2 + (y-b)^2 + c^2 = (x-t*st*cp)^2 + (y-t*st*sp)^2 + c^2
// This gets derived by t to obtain
//    dd^2/dt = -2*st*cp*(x-t*st*cp) -2*st*sp*(y-t*st*sp) +2*t * ct^2 = 0
// which can be solved for t to get
//    t = st*cp*x + st*sp*y
// Now we can compute d^2_min as
//    d^2_min = (x-t*st*cp)^2 + (y-t*st*sp)^2 + (t*ct)^2
// -------------------------------------------------------------------------------------------------------------
double EffectiveDistance (double xd, double yd, double x0, double y0, double theta, double phi, int mode) {
    double dx = xd-x0;
    double dy = yd-y0;
    double r;
    // We treat separately the case of orthogonal showers, which is much easier and faster
    // -----------------------------------------------------------------------------------
    if (theta==0.) {
        r = dx*dx+dy*dy;
        if (r>0.) r = sqrt(r);
        if (r<Rmin) r = Rmin;
        if (mode==0) {
            return r;
        } else if (mode==1) { // Derivative wrt x0
            return -dx/r;
        } else if (mode==2) { // Derivative wrt y0
            return -dy/r;
        } else if (mode==3 || mode==32) { // Derivative wrt theta or d2/dtheta2
            return 0.;
        } else if (mode==4) { // Derivative wrt phi 
            return 0.;
        }
    } else {
        // NB below, the derivative wrt x0, y0 is very easy to compute using R^2 = dx^2 + dy^2 - t^2 
        // and remembering that t depends on x,y so r^2 = dx^2 + dy^2 -(dx*st*cp+dy*st*sp)^2
        // and we get the expressions reported below.
        // -----------------------------------------------------------------------------------------
        double t = sin(theta)*cos(phi)*dx + sin(theta)*sin(phi)*dy;
        r = dx*dx + dy*dy - t*t;
        if (r>0.)   r = sqrt(r);
        if (r<Rmin) r = Rmin;
        if (mode==0) {
            return r;
        } else if (mode==1) { // Derivative wrt x0 (NNBB not x! Minus sign involved). 
            double sp = sin(phi);
            double cp = cos(phi);
            double st = sin(theta);
            double t = st*cp*dx + st*sp*dy;
            return -1./r * (dx-t*st*cp);
        } else if (mode==2) { // Derivative wrt y0 (NNBB not y! Minus sign involved)
            double sp = sin(phi);
            double cp = cos(phi);
            double st = sin(theta);
            double t  = st*cp*dx + st*sp*dy;
            return -1./r * (dy-t*st*sp);
        } else if (mode==3) { // Derivative wrt theta
            // Derived using again R^2 = dx^2 + dy^2 - t^2, where t is solution of min distance
            // --------------------------------------------------------------------------------
            double sp = sin(phi);
            double cp = cos(phi);
            double st = sin(theta);
            double ct = cos(theta);
            double t  = st*cp*dx + st*sp*dy;
            return -t*ct* (dx*cp + dy*sp) / r;
        } else if (mode==32) { // Second derivative wrt theta
            double sp = sin(phi);
            double cp = cos(phi);
            double st = sin(theta);
            double ct = cos(theta);
            double t  = st*cp*dx + st*sp*dy;
            double dtdth   = ct*(dx*cp+dy*sp);
            double d2tdth2 = -st*(dx*cp+dy*sp);
            double d2rdth2 = pow(t*dtdth,2.)/pow(r,1.5) - dtdth*dtdth/r - t*d2tdth2/r;
            return d2rdth2;
        } else if (mode==4) { // Derivative wrt phi
            // Again simpler to use the above expression for r, getting:
            // ---------------------------------------------------------
            double sp = sin(phi);
            double cp = cos(phi);
            double st = sin(theta);
            double ct = cos(theta);
            double t  = st*cp*dx + st*sp*dy;
            return -t*st * (-dx*sp + dy*cp) / r;
        }
    }
    cout    << "     Something fishy in effective distance " << endl;
    outfile << "     Something fishy in effective distance " << endl;
#if defined(STANDALONE) || defined(UBUNTU)
    datamutex.lock();
#endif
    warnings6++;
#if defined(STANDALONE) || defined(UBUNTU)
    datamutex.unlock();
#endif
    return Rmin; // This should not happen
}

// The time of arrival on the ground depends on the variable t of the EffectiveDistance calculation above,
// and it has the correct sign (t>0 for later arrivals) with the definition of t there.
// -------------------------------------------------------------------------------------------------------
double EffectiveTime (double x, double y, double x0, double y0, double theta, double phi, int mode) {
    if (mode==0) {
        return ((x-x0)*sin(theta)*cos(phi) + (y-y0)*sin(theta)*sin(phi))/c0;
        // Note that we could obtain this, probably more economically, by doing T = sqrt(R^2-R_eff^2)
        // but then we'd still need to know the sign, which requires computing asin()
        // ------------------------------------------------------------------------------------------
    } else if (mode==1) { // Derivative wrt x0
        return -sin(theta)*cos(phi)/c0; // note minus sign
    } else if (mode==2) { // Derivative wrt y0
        return -sin(theta)*sin(phi)/c0; // note minus sign
    } else if (mode==3) { // Derivative wrt theta
        return (cos(theta)*((x-x0)*cos(phi) + (y-y0)*sin(phi)))/c0;
    } else if (mode==4) { // Derivative wrt phi
        return (sin(theta)*(-(x-x0)*sin(phi) + (y-y0)*cos(phi)))/c0;
    } 
    return 0.; 
}

// This routine samples the average time and computes its uncertainty (with the avg. time assumed Gaussian-distributed)
// given a number N_exp of particles, a fraction F_bgr of which have a uniform distribution in 
// [T_exp-IntegrationWindow/2, T_exp+IntegrationWindow/2], and the rest have a Gaussian distribution centered in T_exp 
// and of width sigma_time.
// --------------------------------------------------------------------------------------------------------------------
std::pair<double,double> TimeAverage_SigPlusBgr (double T_exp, float Ns, float Nb, bool Obs) {
    // Deal with background first: we have Nb particles sampled from a Uniform distribution
    // so we need to decide what PDF we need to sample Tave_bgr from
    // ------------------------------------------------------------------------------------
    // Even if we know how many particles we actually observed, we do not use that bit in determining
    // the expected time measured by a tank, relying instead on expected values of signal and bacground
    // ------------------------------------------------------------------------------------------------
    double AvTbgr;
    double S_Tbgr;
    int Nbgr, Nsig;
    if (!Obs) {
        Nbgr = myRNG->Poisson(Nb);
        Nsig = myRNG->Poisson(Ns);
    } else {
        Nbgr = Nb;
        Nsig = Ns;
    }
    double tmin = T_exp-IntegrationWindow*0.5;
    double tmax = T_exp+IntegrationWindow*0.5;
    if (Nbgr<=1) {
        // Sample uniformly
        // ----------------
        AvTbgr = myRNG->Uniform(tmin, tmax);
        S_Tbgr = IntegrationWindow/sqrt12;
    } else if (Nbgr<=2) {
        double trial1 = myRNG->Uniform(tmin, tmax);
        double trial2 = myRNG->Uniform(tmin, tmax);
        AvTbgr = 0.5*(trial1+trial2);
        S_Tbgr = IntegrationWindow*.2041; 
    } else if (Nbgr<=3) {
        double trial1 = myRNG->Uniform(tmin, tmax);
        double trial2 = myRNG->Uniform(tmin, tmax);
        double trial3 = myRNG->Uniform(tmin, tmax);
        AvTbgr = (trial1+trial2+trial3)/3.;
        S_Tbgr = IntegrationWindow*0.16666;
    } else if (Nbgr<=4) {
        double trial1 = myRNG->Uniform(tmin, tmax);
        double trial2 = myRNG->Uniform(tmin, tmax);
        double trial3 = myRNG->Uniform(tmin, tmax);
        double trial4 = myRNG->Uniform(tmin, tmax);
        AvTbgr = (trial1+trial2+trial3+trial4)/4.;
        S_Tbgr = IntegrationWindow*0.1445;
    } else { // For N>=5 it is basically Gaussian
        S_Tbgr = IntegrationWindow*0.11; // TBD: Check if 0.11 is ok
        do {
            AvTbgr = myRNG->Gaus(T_exp,S_Tbgr);
        } while (AvTbgr-T_exp>0.5*IntegrationWindow);
    }
    double AvTsig;
    double S_Tsig = sigma_time;
    if (Nsig>=2) S_Tsig = sigma_time/sqrt(-1.+Nsig);
    AvTsig = myRNG->Gaus(T_exp,S_Tsig);
    double mean, rms;
    if (Nbgr==0 && Nsig==0) {
        mean = T_exp;
        rms  = IntegrationWindow/sqrt12;
    } else if (Nbgr==0) {
        mean = AvTsig;
        rms  = S_Tsig;
    } else if (Nsig==0) {
        mean = AvTbgr;
        rms  = S_Tbgr;
    } else {
        double V_Tbgr = S_Tbgr*S_Tbgr;
        double V_Tsig = S_Tsig*S_Tsig;
        double var = sqrt(1./V_Tbgr+1./V_Tsig);
        mean = (AvTbgr/V_Tbgr + AvTsig/V_Tsig)/var;
        rms = sqrt(var);
    }
    return std::make_pair(mean,rms);
} 

// Inverse of 4x4 matrix necessary for interpolation of cubic
// ----------------------------------------------------------
void InitInverse4by4 () {
    //double a11 = A[0][0];
    //double a12 = A[0][1];
    //double a13 = A[0][2];
    //double a14 = A[0][3];
    //double a21 = A[1][0];
    //double a22 = A[1][1];
    //double a23 = A[1][2];
    //double a24 = A[1][3];
    //double a31 = A[2][0];
    //double a32 = A[2][1];
    //double a33 = A[2][2];
    //double a34 = A[2][3];
    //double a41 = A[3][0];
    //double a42 = A[3][1];
    //double a43 = A[3][2];
    //double a44 = A[3][3];
     double a11 = 1;
     double a12 = 1;
     double a13 = 1;
     double a14 = 1;
     double a21 = 1;
     double a22 = 2;
     double a23 = 4;
     double a24 = 8;
     double a31 = 1;
     double a32 = 3;
     double a33 = 9;
     double a34 = 27;
     double a41 = 1;
     double a42 = 4;
     double a43 = 16;
     double a44 = 64;

    // See https://semath.info/src/inverse-cofactor-ex4.html for the details of this explicit calculation
    // of the inverse of a 4x4 matrix. We use the explicit form to speed up calculations
    // --------------------------------------------------------------------------------------------------
    detA = a11*a22*a33*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 -
           a11*a24*a33*a42 - a11*a23*a32*a44 - a11*a22*a34*a43 - 
           a12*a21*a33*a44 - a13*a21*a34*a42 - a14*a21*a32*a43 +
           a14*a21*a33*a42 + a13*a21*a32*a44 + a12*a21*a34*a43 +
           a12*a23*a31*a44 + a13*a24*a31*a42 + a14*a22*a31*a43 -
           a14*a23*a31*a42 - a13*a22*a31*a44 - a12*a24*a31*a43 -
           a12*a23*a34*a41 - a13*a24*a32*a41 - a14*a22*a33*a41 +
           a14*a23*a32*a41 + a13*a22*a34*a41 + a12*a24*a33*a41;

    Ai[0][0] =   a22*a33*a44 + a23*a34*a42 + a24*a32*a43 - 
                 a24*a33*a42 - a23*a32*a44 - a22*a34*a43;
    Ai[0][1] = - a12*a33*a44 - a13*a34*a42 - a14*a32*a43 +
                 a14*a33*a42 + a13*a32*a44 + a12*a34*a43;
    Ai[0][2] =   a12*a23*a44 + a13*a24*a42 + a14*a22*a43 -
                 a14*a23*a42 - a13*a22*a44 - a12*a24*a43;
    Ai[0][3] = - a12*a23*a34 - a13*a24*a32 - a14*a22*a33 +
                 a14*a23*a32 + a13*a22*a34 + a12*a24*a33;

    Ai[1][0] = - a21*a33*a44 - a23*a34*a41 - a24*a31*a43 +
                 a24*a33*a41 + a23*a31*a44 + a21*a34*a43;
    Ai[1][1] =   a11*a33*a44 + a13*a34*a41 + a14*a31*a43 -
                 a14*a33*a41 - a13*a31*a44 - a11*a34*a43;
    Ai[1][2] = - a11*a23*a44 - a13*a24*a41 - a14*a21*a43 +
                 a14*a23*a41 + a13*a21*a44 + a11*a24*a43;
    Ai[1][3] =   a11*a23*a34 + a13*a24*a31 + a14*a21*a33 -
                 a14*a23*a31 - a13*a21*a34 - a11*a24*a33;

    Ai[2][0] =   a21*a32*a44 + a22*a34*a41 + a24*a31*a42 -
                 a24*a32*a41 - a22*a31*a44 - a21*a34*a42;
    Ai[2][1] = - a11*a32*a44 - a12*a34*a41 - a14*a31*a42 +
                 a14*a32*a41 + a12*a31*a44 + a11*a34*a42;
    Ai[2][2] =   a11*a22*a44 + a12*a24*a41 + a14*a21*a42 -
                 a14*a22*a41 - a12*a21*a44 - a11*a24*a42;
    Ai[2][3] = - a11*a22*a34 - a12*a24*a31 - a14*a21*a32 +
                 a14*a22*a31 + a12*a21*a34 + a11*a24*a32;
                 
    Ai[3][0] = - a21*a32*a43 - a22*a33*a41 - a23*a31*a42 +
                 a23*a32*a41 + a22*a31*a43 + a21*a33*a42;
    Ai[3][1] =   a11*a32*a43 + a12*a33*a41 + a13*a31*a42 - 
                 a13*a32*a41 - a12*a31*a43 - a11*a33*a42;
    Ai[3][2] = - a11*a22*a43 - a12*a23*a41 - a13*a21*a42 +
                 a13*a22*a41 + a12*a21*a43 + a11*a23*a42;
    Ai[3][3] =   a11*a22*a33 + a12*a23*a31 + a13*a21*a32 -
                 a13*a22*a31 - a12*a21*a33 - a11*a23*a32;

    for (int i=0; i<4; i++) {
        for (int j=0; j<4; j++) {
            Ai[i][j] /= detA;
        }
    }
    return;
}

// Obtain the four parameters of a cubic passing through the four data points
// (1.,Y[0]), (2.,Y[1]), (3.,Y[2]), (4.,Y[3]) by multiplying the inverse matrix
// by the vector of known Y. 
// This is used to interpolate the parameters of fluxes as a function of theta,
// for fixed energy values. We have 4 theta values at which we evaluate the
// parameters, and we interpolate with a cubic among them.
// ----------------------------------------------------------------------------
void computecubicpars(int mode) {
    for (int i=0; i<4; i++) {
        B[i] = 0.;
    }
    if (mode==0) {
        for (int i=0; i<4; i++) {
            B[0] += Ai[0][i]*Y[i];
            B[1] += Ai[1][i]*Y[i];
            B[2] += Ai[2][i]*Y[i];
            B[3] += Ai[3][i]*Y[i];
        }
    } else if (mode==1) { // Derivatives with respect to Y[]. Here dBdY[i][j] is dB[i]/dY[j]
        for (int i=0; i<4; i++) {
            dBdY[0][i] = Ai[0][i];
            dBdY[1][i] = Ai[1][i];
            dBdY[2][i] = Ai[2][i];
            dBdY[3][i] = Ai[3][i];
        }
    } else {
        cout << "Invalid choice in computecubicpars." << endl;
    }
    return;
}               

// This function obtains the value of parameters thisp0, thisp2 for muons from gammas, or derivatives
// --------------------------------------------------------------------------------------------------
double solvecubic_mg (int parnumber, double energy, double theta, int mode) {

    if (mode==0) { // Primal value
        computecubicpars (0); // This will compute B[] given Y[]
        // Now B[] contains the four parameters of the cubic
        double val = 0.;
        double x = 0.5 + 4. * theta/thetamax;
        for (int i=0; i<4; i++) {
            val += B[i] * pow(x,i);
        }
        return val;
    } else if (mode==2 || mode==22 || mode==25) { 
        // Get 1st, 2nd or 3rd derivative wrt energy 
        // In the above calculation, the value of the cubic at the four points
        // depends on energy through their parametrization, which is different
        // for the three parameters thisp0, thisp1, thisp2.
        // B are the parameters of the cubic that interpolates the Y points:
        //    thisp0 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        // For thisp0, the dependence of the four values is
        //    Y[0] = exp(PXmg1[00]) + exp(PXmg1[01]*pow(f(e),PXmg1[02])) 
        //    Y[1] = exp(PXmg2[00]) + exp(PXmg2[01]*pow(f(e),PXmg2[02])) 
        //    Y[2] = exp(PXmg3[00]) + exp(PXmg3[01]*pow(f(e),PXmg3[02])) 
        //    Y[3] = exp(PXmg4[00]) + exp(PXmg4[01]*pow(f(e),PXmg4[02])) 
        // with 
        //    f(e)  = 0.5 + 20*(log(e)-log01)/(log10-log01)
        //    f'(e) = 20/(log10-log01)/e
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * PXmg1[01]*(PXmg1[02])*f(e)^(PXmg1[02]-1)*exp{}*f'(e) +
        //               dBdY[0][1] * PXmg2[01]*(PXmg2[02])*f(e)^(PXmg2[02]-1)*exp{}*f'(e) +
        //               dBdY[0][2] * PXmg3[01]*(PXmg3[02])*f(e)^(PXmg3[02]-1)*exp{}*f'(e) +   
        //               dBdY[0][3] * PXmg4[01]*(PXmg4[02])*f(e)^(PXmg4[02]-1)*exp{}*f'(e)   
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp0/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        // ----------------------------------------------------------------------------
        // For thisp1, the dependence of the four values is
        //    thisp1 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        //    Y[0] = PXmg1[10] + PXmg1[11] * f(e) + PXmg1[12] * f(e)^2 
        //    Y[1] = PXmg2[10] + PXmg2[11] * f(e) + PXmg2[12] * f(e)^2 
        //    Y[2] = PXmg3[10] + PXmg3[11] * f(e) + PXmg3[12] * f(e)^2 
        //    Y[3] = PXmg4[10] + PXmg4[11] * f(e) + PXmg4[12] * f(e)^2 
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * (PXmg1[11] * f'(e) + PXmg1[12] * 2*f(e)*f'(e)) +
        //               dBdY[0][1] * (PXmg2[11] * f'(e) + PXmg2[12] * 2*f(e)*f'(e)) +
        //               dBdY[0][2] * (PXmg3[11] * f'(e) + PXmg3[12] * 2*f(e)*f'(e)) +   
        //               dBdY[0][3] * (PXmg4[11] * f'(e) + PXmg4[12] * 2*f(e)*f'(e))    
        // and similarly
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp1/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        //
        // For thisp2, the dependence of the four values is
        //    thisp2 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        //    Y[0] = PXmg1[20] + PXmg1[21] * f(e) + PXmg1[22] * f(e)^2 
        //    Y[1] = PXmg2[20] + PXmg2[21] * f(e) + PXmg2[22] * f(e)^2 
        //    Y[2] = PXmg3[20] + PXmg3[21] * f(e) + PXmg3[22] * f(e)^2 
        //    Y[3] = PXmg4[20] + PXmg4[21] * f(e) + PXmg4[22] * f(e)^2 
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * (PXmg1[21] * f'(e) + PXmg1[22] * 2*f(e)*f'(e)) +
        //               dBdY[0][1] * (PXmg2[21] * f'(e) + PXmg2[22] * 2*f(e)*f'(e)) +
        //               dBdY[0][2] * (PXmg3[21] * f'(e) + PXmg3[22] * 2*f(e)*f'(e)) +   
        //               dBdY[0][3] * (PXmg4[21] * f'(e) + PXmg4[22] * 2*f(e)*f'(e))    
        // and similarly
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        //
        // This is the calculation of the four cubic parameters dBdY[][]
        // note: d2B/dy2 terms are all zero (see definition of dBdY[])
        // -------------------------------------------------------------
        computecubicpars (1); // <-- this will compute B[] given Y[]

        // Now all ingredient for dthisp0/de, dthisp1/de, dthisp2/de are there
        // -------------------------------------------------------------------
        double f_e       = 0.5 + 20.*(log(energy)-log_01)/logdif;
        double fprime_e  = 20./logdif/energy;
        double fsecond_e = -1.*fprime_e/energy; // -20./(pow(energy,2.)*logdif);
        double fthird_e  = -2.*fsecond_e/energy; // 60./pow(energy,3.)*logdif);
        double x         = 0.5 + 4. * theta/thetamax;
        double dB0de, dB1de, dB2de, dB3de;
        double d2B0de2, d2B1de2, d2B2de2, d2B3de2;
        double d3B0de3, d3B1de3, d3B2de3, d3B3de3;
        if (parnumber==0) {
            //    dthisp0/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * PXmg1[01]*(PXmg1[02]-1)*exp{}*f'(e) +    
            //               dBdY[0][1] * PXmg2[01]*(PXmg2[02]-1)*exp{}*f'(e) +
            //               dBdY[0][2] * PXmg3[01]*(PXmg3[02]-1)*exp{}*f'(e) +   
            //               dBdY[0][3] * PXmg4[01]*(PXmg4[02]-1)*exp{}*f'(e)    
            double f0 = PXmg1_p[0][1]*PXmg1_p[0][2]*pow(f_e,PXmg1_p[0][2]-1.) * exp(PXmg1_p[0][1]*pow(f_e,PXmg1_p[0][2]))*fprime_e; // this is dY[0]/df * df/de
            double f1 = PXmg2_p[0][1]*PXmg2_p[0][2]*pow(f_e,PXmg2_p[0][2]-1.) * exp(PXmg2_p[0][1]*pow(f_e,PXmg2_p[0][2]))*fprime_e;
            double f2 = PXmg3_p[0][1]*PXmg3_p[0][2]*pow(f_e,PXmg3_p[0][2]-1.) * exp(PXmg3_p[0][1]*pow(f_e,PXmg3_p[0][2]))*fprime_e;
            double f3 = PXmg4_p[0][1]*PXmg4_p[0][2]*pow(f_e,PXmg4_p[0][2]-1.) * exp(PXmg4_p[0][1]*pow(f_e,PXmg4_p[0][2]))*fprime_e;
            if (mode==2) {
                dB0de = dBdY[0][0] * f0 +
                        dBdY[0][1] * f1 +
                        dBdY[0][2] * f2 +
                        dBdY[0][3] * f3;
                dB1de = dBdY[1][0] * f0 +
                        dBdY[1][1] * f1 +
                        dBdY[1][2] * f2 +
                        dBdY[1][3] * f3;
                dB2de = dBdY[2][0] * f0 +
                        dBdY[2][1] * f1 +
                        dBdY[2][2] * f2 +
                        dBdY[2][3] * f3;
                dB3de = dBdY[3][0] * f0 +
                        dBdY[3][1] * f1 +
                        dBdY[3][2] * f2 +
                        dBdY[3][3] * f3;
                return dB0de + dB1de*x + dB2de*x*x + dB3de*x*x*x;
            } else if (mode==22) { // Second derivative wrt energy; 
                double df0_de = f0 * ((PXmg1_p[0][2]-1.+PXmg1_p[0][1]*PXmg1_p[0][2]*pow(f_e,PXmg1_p[0][2]))*fprime_e/f_e + fsecond_e/fprime_e);
                double df1_de = f1 * ((PXmg2_p[0][2]-1.+PXmg2_p[0][1]*PXmg2_p[0][2]*pow(f_e,PXmg2_p[0][2]))*fprime_e/f_e + fsecond_e/fprime_e);
                double df2_de = f2 * ((PXmg3_p[0][2]-1.+PXmg3_p[0][1]*PXmg3_p[0][2]*pow(f_e,PXmg3_p[0][2]))*fprime_e/f_e + fsecond_e/fprime_e);
                double df3_de = f3 * ((PXmg4_p[0][2]-1.+PXmg4_p[0][1]*PXmg4_p[0][2]*pow(f_e,PXmg4_p[0][2]))*fprime_e/f_e + fsecond_e/fprime_e);
                d2B0de2 = dBdY[0][0] * df0_de +  // Because the second derivative d2bdy2 is zero, we only have these terms
                          dBdY[0][1] * df1_de +
                          dBdY[0][2] * df2_de +
                          dBdY[0][3] * df3_de;
                d2B1de2 = dBdY[1][0] * df0_de +
                          dBdY[1][1] * df1_de +
                          dBdY[1][2] * df2_de +
                          dBdY[1][3] * df3_de;
                d2B2de2 = dBdY[2][0] * df0_de +
                          dBdY[2][1] * df1_de +
                          dBdY[2][2] * df2_de +
                          dBdY[2][3] * df3_de;
                d2B3de2 = dBdY[3][0] * df0_de +
                          dBdY[3][1] * df1_de +
                          dBdY[3][2] * df2_de +
                          dBdY[3][3] * df3_de;
                return d2B0de2 + d2B1de2*x + d2B2de2*x*x + d2B3de2*x*x*x;
            } else if (mode==25) { // Third derivative wrt energy;
                double p1 = PXmg1_p[0][1];
                double p2 = PXmg1_p[0][2];
                double f_etop2 = pow(f_e,p2);
                double d2f0_de2 = f0/pow(f_e,2.) * ((2.-3.*p2+pow(p2,2.)+3.*p1*(p2-1.)*p2*f_etop2+pow(p1*p2*f_etop2,2.))*pow(fprime_e,2.) 
                                  + 3.*f_e*(-1.+p2+p1*p2*f_etop2)*fsecond_e+pow(f_e,2.)*fthird_e/fprime_e);
                p1 = PXmg2_p[0][1];
                p2 = PXmg2_p[0][2];
                f_etop2 = pow(f_e,p2);
                double d2f1_de2 = f1/pow(f_e,2.) * ((2.-3.*p2+pow(p2,2.)+3.*p1*(p2-1.)*p2*f_etop2+pow(p1*p2*f_etop2,2.))*pow(fprime_e,2.) 
                                  + 3.*f_e*(-1.+p2+p1*p2*f_etop2)*fsecond_e+pow(f_e,2.)*fthird_e/fprime_e);
                p1 = PXmg3_p[0][1];
                p2 = PXmg3_p[0][2];
                f_etop2 = pow(f_e,p2);
                double d2f2_de2 = f2/pow(f_e,2.) * ((2.-3.*p2+pow(p2,2.)+3.*p1*(p2-1.)*p2*f_etop2+pow(p1*p2*f_etop2,2.))*pow(fprime_e,2.) 
                                  + 3.*f_e*(-1.+p2+p1*p2*f_etop2)*fsecond_e+pow(f_e,2.)*fthird_e/fprime_e);
                p1 = PXmg4_p[0][1];
                p2 = PXmg4_p[0][2];
                f_etop2 = pow(f_e,p2);
                double d2f3_de2 = f3/pow(f_e,2.) * ((2.-3.*p2+pow(p2,2.)+3.*p1*(p2-1.)*p2*f_etop2+pow(p1*p2*f_etop2,2.))*pow(fprime_e,2.) 
                                  + 3.*f_e*(-1.+p2+p1*p2*f_etop2)*fsecond_e+pow(f_e,2.)*fthird_e/fprime_e);
                d3B0de3 = dBdY[0][0] * d2f0_de2 +
                          dBdY[0][1] * d2f1_de2 +
                          dBdY[0][2] * d2f2_de2 +
                          dBdY[0][3] * d2f3_de2;
                d3B1de3 = dBdY[1][0] * d2f0_de2 +
                          dBdY[1][1] * d2f1_de2 +
                          dBdY[1][2] * d2f2_de2 +
                          dBdY[1][3] * d2f3_de2;
                d3B2de3 = dBdY[2][0] * d2f0_de2 +
                          dBdY[2][1] * d2f1_de2 +
                          dBdY[2][2] * d2f2_de2 +
                          dBdY[2][3] * d2f3_de2;
                d3B3de3 = dBdY[3][0] * d2f0_de2 +
                          dBdY[3][1] * d2f1_de2 +
                          dBdY[3][2] * d2f2_de2 +
                          dBdY[3][3] * d2f3_de2;
                return d3B0de3 + d3B1de3*x + d3B2de3*x*x + d3B3de3*x*x*x; // WIP
            }
        } else if (parnumber==2) { // No parnumber 1 for muons
            //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * (PXmg1[21] * f'(e) + PXmg1[22] * 2*f(e)*f'(e)) +
            //               dBdY[0][1] * (PXmg2[21] * f'(e) + PXmg2[22] * 2*f(e)*f'(e)) +
            //               dBdY[0][2] * (PXmg3[21] * f'(e) + PXmg3[22] * 2*f(e)*f'(e)) +   
            //               dBdY[0][3] * (PXmg4[21] * f'(e) + PXmg4[22] * 2*f(e)*f'(e))    
            double f0 = PXmg1_p[2][1]*fprime_e + PXmg1_p[2][2]*2.*f_e*fprime_e;
            double f1 = PXmg2_p[2][1]*fprime_e + PXmg2_p[2][2]*2.*f_e*fprime_e;
            double f2 = PXmg3_p[2][1]*fprime_e + PXmg3_p[2][2]*2.*f_e*fprime_e;
            double f3 = PXmg4_p[2][1]*fprime_e + PXmg4_p[2][2]*2.*f_e*fprime_e;
            if (mode==2) { // First derivative wrt energy
                dB0de = dBdY[0][0] * f0 + 
                        dBdY[0][1] * f1 + 
                        dBdY[0][2] * f2 + 
                        dBdY[0][3] * f3;
                dB1de = dBdY[1][0] * f0 + 
                        dBdY[1][1] * f1 + 
                        dBdY[1][2] * f2 + 
                        dBdY[1][3] * f3;
                dB2de = dBdY[2][0] * f0 + 
                        dBdY[2][1] * f1 + 
                        dBdY[2][2] * f2 + 
                        dBdY[2][3] * f3;
                dB3de = dBdY[3][0] * f0 + 
                        dBdY[3][1] * f1 + 
                        dBdY[3][2] * f2 + 
                        dBdY[3][3] * f3;
                return dB0de + dB1de*x + dB2de*x*x + dB3de*x*x*x;
            } else if (mode==22) { // Second derivative wrt energy
                // We are computing d^2thisp2/de2 = d^2B[0]/de2 + d^2B[1]/de2 * xt + ...
                // This is = dBdY[0][0]* d/de [(PXmg1[21] * f'(e) + PXmg1[22] * 2*f(e)*f'(e))] plus the three other terms, see above.
                // ------------------------------------------------------------------------------------------------------------------
                double df0de = (PXmg1_p[2][1]*fsecond_e+2.*PXmg1_p[2][2]*(pow(fprime_e,2)+f_e*fsecond_e));
                double df1de = (PXmg2_p[2][1]*fsecond_e+2.*PXmg2_p[2][2]*(pow(fprime_e,2)+f_e*fsecond_e));
                double df2de = (PXmg3_p[2][1]*fsecond_e+2.*PXmg3_p[2][2]*(pow(fprime_e,2)+f_e*fsecond_e));
                double df3de = (PXmg4_p[2][1]*fsecond_e+2.*PXmg4_p[2][2]*(pow(fprime_e,2)+f_e*fsecond_e));
                d2B0de2 = dBdY[0][0] * df0de +
                          dBdY[0][1] * df1de +
                          dBdY[0][2] * df2de +
                          dBdY[0][3] * df3de;
                d2B1de2 = dBdY[1][0] * df0de +
                          dBdY[1][1] * df1de +
                          dBdY[1][2] * df2de +
                          dBdY[1][3] * df3de;
                d2B2de2 = dBdY[2][0] * df0de +
                          dBdY[2][1] * df1de +
                          dBdY[2][2] * df2de +
                          dBdY[2][3] * df3de;
                d2B3de2 = dBdY[3][0] * df0de +
                          dBdY[3][1] * df1de +
                          dBdY[3][2] * df2de +
                          dBdY[3][3] * df3de;
                return d2B0de2 + d2B1de2*x + d2B2de2*x*x + d2B3de2*x*x*x;
            } else if (mode==25) { // Third derivative wrt energy;
                double d2f0_de2 = PXmg1_p[2][1]*fthird_e+2.*PXmg1_p[2][2]*(3.*fprime_e*fsecond_e+f_e*fthird_e);
                double d2f1_de2 = PXmg2_p[2][1]*fthird_e+2.*PXmg2_p[2][2]*(3.*fprime_e*fsecond_e+f_e*fthird_e);
                double d2f2_de2 = PXmg3_p[2][1]*fthird_e+2.*PXmg3_p[2][2]*(3.*fprime_e*fsecond_e+f_e*fthird_e);
                double d2f3_de2 = PXmg4_p[2][1]*fthird_e+2.*PXmg4_p[2][2]*(3.*fprime_e*fsecond_e+f_e*fthird_e);
                d3B0de3 = dBdY[0][0] * d2f0_de2 +
                          dBdY[0][1] * d2f1_de2 +
                          dBdY[0][2] * d2f2_de2 +
                          dBdY[0][3] * d2f3_de2;
                d3B1de3 = dBdY[1][0] * d2f0_de2 +
                          dBdY[1][1] * d2f1_de2 +
                          dBdY[1][2] * d2f2_de2 +
                          dBdY[1][3] * d2f3_de2;
                d3B2de3 = dBdY[2][0] * d2f0_de2 +
                          dBdY[2][1] * d2f1_de2 +
                          dBdY[2][2] * d2f2_de2 +
                          dBdY[2][3] * d2f3_de2;
                d3B3de3 = dBdY[3][0] * d2f0_de2 +
                          dBdY[3][1] * d2f1_de2 +
                          dBdY[3][2] * d2f2_de2 +
                          dBdY[3][3] * d2f3_de2;
                return d3B0de3 + d3B1de3*x + d3B2de3*x*x + d3B3de3*x*x*x; // WIP
            }
        }
    } else if (mode==3 || mode==32) { // Derivative wrt theta or second derivative wrt theta2
        computecubicpars (0);
        // Now B[] contains the four parameters of the cubic
        double x = 0.5 + 4. * theta/thetamax;
        double dxdtheta = 4./thetamax;
        if (mode==3) {
            return (B[1] + B[2]*2.*x + B[3] * 3.*x*x) * dxdtheta;
        } else if (mode==32) {
            return (2.*B[2] + 6.*B[3]*x)*dxdtheta; // no d2xdtheta2 term, null
        }
    } 
    // If it gets here an invalid value of mode was passed
    // ---------------------------------------------------
    cout    << "Warning - invalid mode for cubic in solvecubic_mg" << endl;
    outfile << "Warning - invalid mode for cubic in solvecubic_mg" << endl;
    warnings1++;
    return 0.; // This should not happen
}

// This function obtains the value of parameters thisp0, thisp1, thisp2 for e+g from gammas, or derivatives
// --------------------------------------------------------------------------------------------------------
double solvecubic_eg (int parnumber, double energy, double theta, int mode) {

    if (mode==0) { // Primal value

        computecubicpars (0); // This will compute B[] given Y[]
        // Now B[] contains the four parameters of the cubic
        double val = 0.;
        double x = 0.5 + 4. * theta/thetamax;
        for (int i=0; i<4; i++) {
            val += B[i] * pow(x,i);
        }
        return val;
    } else if (mode==2 || mode==22 || mode==25) { // Get first or second derivative wrt energy
        // In the above calculation, the value of the cubic at the four points
        // depends on energy through their parametrization, which is different
        // for the three parameters thisp0, thisp1, thisp2.
        //    thisp0 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        // For thisp0, the dependence of the four values is
        //    Y[0] = PXeg1[00] * exp(PXeg1[01]*pow(f(e),PXeg1[02])) 
        //    Y[1] = PXeg2[00] * exp(PXeg2[01]*pow(f(e),PXeg2[02])) 
        //    Y[2] = PXeg3[00] * exp(PXeg3[01]*pow(f(e),PXeg3[02])) 
        //    Y[3] = PXeg4[00] * exp(PXeg4[01]*pow(f(e),PXeg4[02])) 
        // with 
        //    f(e)  = 0.5 + 20*(log(e)-log01)/(log10-log01)
        //    f'(e) = 20/(log10-log01)/e
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * PXeg1[00]*PXeg1[01]*PXeg1[02]*f_e^(PXeg1[02]-1)*exp{}*f'(e) +
        //               dBdY[0][1] * PXeg2[00]*PXeg2[01]*PXeg2[02]*f_e^(PXeg2[02]-1)*exp{}*f'(e) +
        //               dBdY[0][2] * PXeg3[00]*PXeg3[01]*PXeg3[02]*f_e^(PXeg3[02]-1)*exp{}*f'(e) +
        //               dBdY[0][3] * PXeg4[00]*PXeg4[01]*PXeg4[02]*f_e^(PXeg4[02]-1)*exp{}*f'(e) 
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp0/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        // ----------------------------------------------------------------------------
        // For thisp1, the dependence of the four values is
        //    thisp1 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        //    Y[0] = PXeg1[10] + PXeg1[11] * f(e) + PXeg1[12] * f(e)^2 
        //    Y[1] = PXeg2[10] + PXeg2[11] * f(e) + PXeg2[12] * f(e)^2 
        //    Y[2] = PXeg3[10] + PXeg3[11] * f(e) + PXeg3[12] * f(e)^2 
        //    Y[3] = PXeg4[10] + PXeg4[11] * f(e) + PXeg4[12] * f(e)^2 
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * (PXeg1[11] * f'(e) + PXeg1[12] * 2*f(e)*f'(e)) +
        //               dBdY[0][1] * (PXeg2[11] * f'(e) + PXeg2[12] * 2*f(e)*f'(e)) +
        //               dBdY[0][2] * (PXeg3[11] * f'(e) + PXeg3[12] * 2*f(e)*f'(e)) +   
        //               dBdY[0][3] * (PXeg4[11] * f'(e) + PXeg4[12] * 2*f(e)*f'(e))    
        // and similarly
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp1/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        //
        // For thisp2, the dependence of the four values is
        //    thisp2 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        //    Y[0] = PXeg1[20] + PXeg1[21] * f(e) + PXeg1[22] * f(e)^2 
        //    Y[1] = PXeg2[20] + PXeg2[21] * f(e) + PXeg2[22] * f(e)^2 
        //    Y[2] = PXeg3[20] + PXeg3[21] * f(e) + PXeg3[22] * f(e)^2 
        //    Y[3] = PXeg4[20] + PXeg4[21] * f(e) + PXeg4[22] * f(e)^2 
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * (PXeg1[21] * f'(e) + PXeg1[22] * 2*f(e)*f'(e)) +
        //               dBdY[0][1] * (PXeg2[21] * f'(e) + PXeg2[22] * 2*f(e)*f'(e)) +
        //               dBdY[0][2] * (PXeg3[21] * f'(e) + PXeg3[22] * 2*f(e)*f'(e)) +   
        //               dBdY[0][3] * (PXeg4[21] * f'(e) + PXeg4[22] * 2*f(e)*f'(e))    
        // and similarly
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        // 
        // This is the calculation of the four cubic parameters B[]
        // --------------------------------------------------------
        computecubicpars (1); // This will compute B[] given Y[]

        // Now all ingredient for dthisp0/de, dthisp1/de, dthisp2/de are there
        // -------------------------------------------------------------------
        double f_e       = 0.5 + 20.*(log(energy)-log_01)/logdif;
        double fprime_e  = 20./logdif/energy;
        double fsecond_e = -1.*fprime_e/energy; // -20./(pow(energy,2.)*logdif);
        double fthird_e  = -2.*fsecond_e/energy;
        double x         = 0.5 + 4. * theta/thetamax;
        double dB0de, dB1de, dB2de, dB3de;
        double d2B0de2, d2B1de2, d2B2de2, d2B3de2;
        double d3B0de3, d3B1de3, d3B2de3, d3B3de3;
        if (parnumber==0) {
            //    dthisp0/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * PXeg1[00]*PXeg1[01]*(PXeg1[02]-1)*exp{}*f'(e) +
            //               dBdY[0][1] * PXeg2[00]*PXeg2[01]*(PXeg2[02]-1)*exp{}*f'(e) +
            //               dBdY[0][2] * PXeg3[00]*PXeg3[01]*(PXeg3[02]-1)*exp{}*f'(e) +   
            //               dBdY[0][3] * PXeg4[00]*PXeg4[01]*(PXeg4[02]-1)*exp{}*f'(e)    
            double f0 = PXeg1_p[0][0]*PXeg1_p[0][1]*PXeg1_p[0][2]*pow(f_e,PXeg1_p[0][2]-1.)*exp(PXeg1_p[0][1]*pow(f_e,PXeg1_p[0][2]))*fprime_e;
            double f1 = PXeg2_p[0][0]*PXeg2_p[0][1]*PXeg2_p[0][2]*pow(f_e,PXeg2_p[0][2]-1.)*exp(PXeg2_p[0][1]*pow(f_e,PXeg2_p[0][2]))*fprime_e;
            double f2 = PXeg3_p[0][0]*PXeg3_p[0][1]*PXeg3_p[0][2]*pow(f_e,PXeg3_p[0][2]-1.)*exp(PXeg3_p[0][1]*pow(f_e,PXeg3_p[0][2]))*fprime_e;
            double f3 = PXeg4_p[0][0]*PXeg4_p[0][1]*PXeg4_p[0][2]*pow(f_e,PXeg4_p[0][2]-1.)*exp(PXeg4_p[0][1]*pow(f_e,PXeg4_p[0][2]))*fprime_e;
            if (mode==2) {
                dB0de = dBdY[0][0] * f0 +
                        dBdY[0][1] * f1 +
                        dBdY[0][2] * f2 +
                        dBdY[0][3] * f3;
                dB1de = dBdY[1][0] * f0 +
                        dBdY[1][1] * f1 +
                        dBdY[1][2] * f2 +
                        dBdY[1][3] * f3;
                dB2de = dBdY[2][0] * f0 +
                        dBdY[2][1] * f1 +
                        dBdY[2][2] * f2 +
                        dBdY[2][3] * f3;
                dB3de = dBdY[3][0] * f0 +
                        dBdY[3][1] * f1 +
                        dBdY[3][2] * f2 +
                        dBdY[3][3] * f3;
                return dB0de + dB1de*x + dB2de*x*x + dB3de*x*x*x;
            } else if (mode==22) { // Second derivative wrt energy
                double df0_de = f0 * ((PXeg1_p[0][2]-1.)*fprime_e/f_e + PXeg1_p[0][1]*PXeg1_p[0][2]*pow(f_e,PXeg1_p[0][2]-1.)*fprime_e  + fsecond_e/fprime_e);
                double df1_de = f1 * ((PXeg2_p[0][2]-1.)*fprime_e/f_e + PXeg2_p[0][1]*PXeg2_p[0][2]*pow(f_e,PXeg2_p[0][2]-1.)*fprime_e  + fsecond_e/fprime_e);
                double df2_de = f2 * ((PXeg3_p[0][2]-1.)*fprime_e/f_e + PXeg3_p[0][1]*PXeg3_p[0][2]*pow(f_e,PXeg3_p[0][2]-1.)*fprime_e  + fsecond_e/fprime_e);
                double df3_de = f3 * ((PXeg4_p[0][2]-1.)*fprime_e/f_e + PXeg4_p[0][1]*PXeg4_p[0][2]*pow(f_e,PXeg4_p[0][2]-1.)*fprime_e  + fsecond_e/fprime_e);
                d2B0de2 = dBdY[0][0] * df0_de +
                          dBdY[0][1] * df1_de +
                          dBdY[0][2] * df2_de +
                          dBdY[0][3] * df3_de;
                d2B1de2 = dBdY[1][0] * df0_de +
                          dBdY[1][1] * df1_de +
                          dBdY[1][2] * df2_de +
                          dBdY[1][3] * df3_de;
                d2B2de2 = dBdY[2][0] * df0_de +
                          dBdY[2][1] * df1_de +
                          dBdY[2][2] * df2_de +
                          dBdY[2][3] * df3_de;
                d2B3de2 = dBdY[3][0] * df0_de +
                          dBdY[3][1] * df1_de +
                          dBdY[3][2] * df2_de +
                          dBdY[3][3] * df3_de;
                return d2B0de2 + d2B1de2*x + d2B2de2*x*x + d2B3de2*x*x*x;
            } else if (mode==25) { // Second derivative wrt energy
                double p1 = PXeg1_p[0][1];
                double p2 = PXeg1_p[0][2];
                double f_etop2 = pow(f_e,p2);
                double d2f0_de2 = f0 /(fprime_e*pow(p2,2.)) * ((2.-3.*p2+p2*p2+3.*p1*(p2-1.)*p2*pow(f_e,p2)+pow(p1*p2*f_etop2,2.))*pow(fprime_e,3.)
                                  + 3.*f_e * (-1.+p2+p1*p2*f_etop2)*fprime_e*fsecond_e+f_e*f_e*fthird_e); 
                p1 = PXeg2_p[0][1];
                p2 = PXeg2_p[0][2];
                f_etop2 = pow(f_e,p2);
                double d2f1_de2 = f1 /(fprime_e*pow(p2,2.)) * ((2.-3.*p2+p2*p2+3.*p1*(p2-1.)*p2*pow(f_e,p2)+pow(p1*p2*f_etop2,2.))*pow(fprime_e,3.)
                                  + 3.*f_e * (-1.+p2+p1*p2*f_etop2)*fprime_e*fsecond_e+f_e*f_e*fthird_e); 
                p1 = PXeg3_p[0][1];
                p2 = PXeg3_p[0][2];
                f_etop2 = pow(f_e,p2);
                double d2f2_de2 = f2 /(fprime_e*pow(p2,2.)) * ((2.-3.*p2+p2*p2+3.*p1*(p2-1.)*p2*pow(f_e,p2)+pow(p1*p2*f_etop2,2.))*pow(fprime_e,3.)
                                  + 3.*f_e * (-1.+p2+p1*p2*f_etop2)*fprime_e*fsecond_e+f_e*f_e*fthird_e); 
                p1 = PXeg4_p[0][1];
                p2 = PXeg4_p[0][2];
                f_etop2 = pow(f_e,p2);
                double d2f3_de2 = f3 /(fprime_e*pow(p2,2.)) * ((2.-3.*p2+p2*p2+3.*p1*(p2-1.)*p2*pow(f_e,p2)+pow(p1*p2*f_etop2,2.))*pow(fprime_e,3.)
                                  + 3.*f_e * (-1.+p2+p1*p2*f_etop2)*fprime_e*fsecond_e+f_e*f_e*fthird_e); 
                d3B0de3 = dBdY[0][0] * d2f0_de2 +
                          dBdY[0][1] * d2f1_de2 +
                          dBdY[0][2] * d2f2_de2 +
                          dBdY[0][3] * d2f3_de2;
                d3B1de3 = dBdY[1][0] * d2f0_de2 +
                          dBdY[1][1] * d2f1_de2 +
                          dBdY[1][2] * d2f2_de2 +
                          dBdY[1][3] * d2f3_de2;
                d3B2de3 = dBdY[2][0] * d2f0_de2 +
                          dBdY[2][1] * d2f1_de2 +
                          dBdY[2][2] * d2f2_de2 +
                          dBdY[2][3] * d2f3_de2;
                d3B3de3 = dBdY[3][0] * d2f0_de2 +
                          dBdY[3][1] * d2f1_de2 +
                          dBdY[3][2] * d2f2_de2 +
                          dBdY[3][3] * d2f3_de2;
                return d3B0de3 + d3B1de3*x + d3B2de3*x*x + d3B3de3*x*x*x;
            }
        } else if (parnumber==1) {  
            //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * (PXeg1[11] * f'(e) + PXeg1[12] * 2*f(e)*f'(e)) +
            //               dBdY[0][1] * (PXeg2[11] * f'(e) + PXeg2[12] * 2*f(e)*f'(e)) +
            //               dBdY[0][2] * (PXeg3[11] * f'(e) + PXeg3[12] * 2*f(e)*f'(e)) +   
            //               dBdY[0][3] * (PXeg4[11] * f'(e) + PXeg4[12] * 2*f(e)*f'(e))    
            double f1 = PXeg1_p[1][1]*fprime_e + PXeg1_p[1][2]*2.*f_e*fprime_e;
            double f2 = PXeg2_p[1][1]*fprime_e + PXeg2_p[1][2]*2.*f_e*fprime_e;
            double f3 = PXeg3_p[1][1]*fprime_e + PXeg3_p[1][2]*2.*f_e*fprime_e;
            double f4 = PXeg4_p[1][1]*fprime_e + PXeg4_p[1][2]*2.*f_e*fprime_e;
            if (mode==2) { 
                dB0de = dBdY[0][0] * f1 + 
                        dBdY[0][1] * f2 + 
                        dBdY[0][2] * f3 + 
                        dBdY[0][3] * f4;
                dB1de = dBdY[1][0] * f1 + 
                        dBdY[1][1] * f2 + 
                        dBdY[1][2] * f3 + 
                        dBdY[1][3] * f4;
                dB2de = dBdY[2][0] * f1 + 
                        dBdY[2][1] * f2 + 
                        dBdY[2][2] * f3 + 
                        dBdY[2][3] * f4;
                dB3de = dBdY[3][0] * f1 + 
                        dBdY[3][1] * f2 + 
                        dBdY[3][2] * f3 + 
                        dBdY[3][3] * f4;
                return dB0de + dB1de*x + dB2de*x*x + dB3de*x*x*x;
            } else if (mode==22) {
                double df1_de = PXeg1_p[1][1]*fsecond_e + PXeg1_p[1][2]*2.*(pow(fprime_e,2.)+f_e*fsecond_e);
                double df2_de = PXeg2_p[1][1]*fsecond_e + PXeg2_p[1][2]*2.*(pow(fprime_e,2.)+f_e*fsecond_e);
                double df3_de = PXeg3_p[1][1]*fsecond_e + PXeg3_p[1][2]*2.*(pow(fprime_e,2.)+f_e*fsecond_e);
                double df4_de = PXeg4_p[1][1]*fsecond_e + PXeg4_p[1][2]*2.*(pow(fprime_e,2.)+f_e*fsecond_e);
                d2B0de2 = dBdY[0][0] * df1_de + 
                          dBdY[0][1] * df2_de + 
                          dBdY[0][2] * df3_de + 
                          dBdY[0][3] * df4_de;
                d2B1de2 = dBdY[1][0] * df1_de + 
                          dBdY[1][1] * df2_de + 
                          dBdY[1][2] * df3_de + 
                          dBdY[1][3] * df4_de;
                d2B2de2 = dBdY[2][0] * df1_de + 
                          dBdY[2][1] * df2_de + 
                          dBdY[2][2] * df3_de + 
                          dBdY[2][3] * df4_de;
                d2B3de2 = dBdY[3][0] * df1_de + 
                          dBdY[3][1] * df2_de + 
                          dBdY[3][2] * df3_de + 
                          dBdY[3][3] * df4_de;
                return d2B0de2 + d2B1de2*x + d2B2de2*x*x + d2B3de2*x*x*x;
            } else if (mode==25) { 
                double d2f1_de2 = PXeg1_p[1][1]*fthird_e + PXeg1_p[1][2]*2.*(3.*fprime_e*fsecond_e +f_e*fthird_e);
                double d2f2_de2 = PXeg2_p[1][1]*fthird_e + PXeg2_p[1][2]*2.*(3.*fprime_e*fsecond_e +f_e*fthird_e);
                double d2f3_de2 = PXeg3_p[1][1]*fthird_e + PXeg3_p[1][2]*2.*(3.*fprime_e*fsecond_e +f_e*fthird_e);
                double d2f4_de2 = PXeg4_p[1][1]*fthird_e + PXeg4_p[1][2]*2.*(3.*fprime_e*fsecond_e +f_e*fthird_e);
                d2B0de2 = dBdY[0][0] * d2f1_de2 + 
                          dBdY[0][1] * d2f2_de2 + 
                          dBdY[0][2] * d2f3_de2 + 
                          dBdY[0][3] * d2f4_de2;
                d2B1de2 = dBdY[1][0] * d2f1_de2 + 
                          dBdY[1][1] * d2f2_de2 + 
                          dBdY[1][2] * d2f3_de2 + 
                          dBdY[1][3] * d2f4_de2;
                d2B2de2 = dBdY[2][0] * d2f1_de2 + 
                          dBdY[2][1] * d2f2_de2 + 
                          dBdY[2][2] * d2f3_de2 + 
                          dBdY[2][3] * d2f4_de2;
                d2B3de2 = dBdY[3][0] * d2f1_de2 + 
                          dBdY[3][1] * d2f2_de2 + 
                          dBdY[3][2] * d2f3_de2 + 
                          dBdY[3][3] * d2f4_de2;
                return d2B0de2 + d2B1de2*x + d2B2de2*x*x + d2B3de2*x*x*x;
            }
        } else if (parnumber==2) {  
            //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * (PXeg1[21] * f'(e) + PXeg1[22] * 2*f(e)*f'(e)) +
            //               dBdY[0][1] * (PXeg2[21] * f'(e) + PXeg2[22] * 2*f(e)*f'(e)) +
            //               dBdY[0][2] * (PXeg3[21] * f'(e) + PXeg3[22] * 2*f(e)*f'(e)) +   
            //               dBdY[0][3] * (PXeg4[21] * f'(e) + PXeg4[22] * 2*f(e)*f'(e))    
            double f1 = PXeg1_p[2][1]*fprime_e + PXeg1_p[2][2]*2.*f_e*fprime_e;
            double f2 = PXeg2_p[2][1]*fprime_e + PXeg2_p[2][2]*2.*f_e*fprime_e;
            double f3 = PXeg3_p[2][1]*fprime_e + PXeg3_p[2][2]*2.*f_e*fprime_e;
            double f4 = PXeg4_p[2][1]*fprime_e + PXeg4_p[2][2]*2.*f_e*fprime_e;
            if (mode==2) {
                dB0de = dBdY[0][0] * f1 + 
                        dBdY[0][1] * f2 + 
                        dBdY[0][2] * f3 + 
                        dBdY[0][3] * f4;
                dB1de = dBdY[1][0] * f1 + 
                        dBdY[1][1] * f2 + 
                        dBdY[1][2] * f3 + 
                        dBdY[1][3] * f4;
                dB2de = dBdY[2][0] * f1 + 
                        dBdY[2][1] * f2 + 
                        dBdY[2][2] * f3 + 
                        dBdY[2][3] * f4;
                dB3de = dBdY[3][0] * f1 + 
                        dBdY[3][1] * f2 + 
                        dBdY[3][2] * f3 + 
                        dBdY[3][3] * f4;
                return dB0de + dB1de*x + dB2de*x*x + dB3de*x*x*x;
            } else if (mode==22) {
                double df1_de = PXeg1_p[2][1]*fsecond_e + PXeg1_p[2][2]*2.*(pow(fprime_e,2.)+f_e*fsecond_e);
                double df2_de = PXeg2_p[2][1]*fsecond_e + PXeg2_p[2][2]*2.*(pow(fprime_e,2.)+f_e*fsecond_e);
                double df3_de = PXeg3_p[2][1]*fsecond_e + PXeg3_p[2][2]*2.*(pow(fprime_e,2.)+f_e*fsecond_e);
                double df4_de = PXeg4_p[2][1]*fsecond_e + PXeg4_p[2][2]*2.*(pow(fprime_e,2.)+f_e*fsecond_e);
                d2B0de2 = dBdY[0][0] * df1_de + 
                          dBdY[0][1] * df2_de + 
                          dBdY[0][2] * df3_de + 
                          dBdY[0][3] * df4_de;
                d2B1de2 = dBdY[1][0] * df1_de + 
                          dBdY[1][1] * df2_de + 
                          dBdY[1][2] * df3_de + 
                          dBdY[1][3] * df4_de;
                d2B2de2 = dBdY[2][0] * df1_de + 
                          dBdY[2][1] * df2_de + 
                          dBdY[2][2] * df3_de + 
                          dBdY[2][3] * df4_de;
                d2B3de2 = dBdY[3][0] * df1_de + 
                          dBdY[3][1] * df2_de + 
                          dBdY[3][2] * df3_de + 
                          dBdY[3][3] * df4_de;
                return d2B0de2 + d2B1de2*x + d2B2de2*x*x + d2B3de2*x*x*x;
            } else if (mode==25) {
                double d2f1_de2 = PXeg1_p[2][1]*fthird_e + PXeg1_p[2][2]*2.*(3.*fprime_e*fsecond_e+f_e*fthird_e);
                double d2f2_de2 = PXeg2_p[2][1]*fthird_e + PXeg2_p[2][2]*2.*(3.*fprime_e*fsecond_e+f_e*fthird_e);
                double d2f3_de2 = PXeg3_p[2][1]*fthird_e + PXeg3_p[2][2]*2.*(3.*fprime_e*fsecond_e+f_e*fthird_e);
                double d2f4_de2 = PXeg4_p[2][1]*fthird_e + PXeg4_p[2][2]*2.*(3.*fprime_e*fsecond_e+f_e*fthird_e);
                d2B0de2 = dBdY[0][0] * d2f1_de2 + 
                          dBdY[0][1] * d2f2_de2 + 
                          dBdY[0][2] * d2f3_de2 + 
                          dBdY[0][3] * d2f4_de2;
                d2B1de2 = dBdY[1][0] * d2f1_de2 + 
                          dBdY[1][1] * d2f2_de2 + 
                          dBdY[1][2] * d2f3_de2 + 
                          dBdY[1][3] * d2f4_de2;
                d2B2de2 = dBdY[2][0] * d2f1_de2 + 
                          dBdY[2][1] * d2f2_de2 + 
                          dBdY[2][2] * d2f3_de2 + 
                          dBdY[2][3] * d2f4_de2;
                d2B3de2 = dBdY[3][0] * d2f1_de2 + 
                          dBdY[3][1] * d2f2_de2 + 
                          dBdY[3][2] * d2f3_de2 + 
                          dBdY[3][3] * d2f4_de2;
                return d2B0de2 + d2B1de2*x + d2B2de2*x*x + d2B3de2*x*x*x;
            }            
        }
    } else if (mode==3 || mode==32) { // Derivative wrt theta or d2/dtheta2
        computecubicpars (0);
        // Now B[] contains the four parameters of the cubic
        double x = 0.5 + 4. * theta/thetamax;
        double dxdtheta = 4./thetamax;
        if (mode==3) {
            return (B[1] + B[2]*2.*x + B[3] * 3.*x*x) * dxdtheta;
        } else if (mode==32) {
            return (2.*B[2] + 6.*B[3]*x)*dxdtheta; // No d2xdtheta2 term, null
        }
    } 
    // If it gets here an invalid value of mode was passed
    // ---------------------------------------------------
    cout    << "Warning - invalid mode for cubic in solvecubic_eg" << endl;
    outfile << "Warning - invalid mode for cubic in solvecubic_eg" << endl;
    warnings1++;
    return 0.; // This should not happen
}

// This function obtains the value of parameters thisp0, thisp2 for muons from protons, or derivatives
// ---------------------------------------------------------------------------------------------------
double solvecubic_mp (int parnumber, double energy, double theta, int mode) {

    if (mode==0) { // Primal value
        computecubicpars (0); // This will compute B[] given Y[]
        // Now B[] contains the four parameters of the cubic
        double val = 0.;
        double x = 0.5 + 4. * theta/thetamax;
        for (int i=0; i<4; i++) {
            val += B[i] * pow(x,i);
        }
        return val;
    } else if (mode==2) { // Get derivative wrt energy
        // In the above calculation, the value of the cubic at the four points
        // depends on energy through their parametrization, which is different
        // for the three parameters thisp0, thisp1, thisp2.
        //    thisp0 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        // For thisp0, the dependence of the four values is
        //    Y[0] = exp(PXmp1[00]) + exp(PXmp1[01]*pow(f(e),PXmp1[02])) 
        //    Y[1] = exp(PXmp2[00]) + exp(PXmp2[01]*pow(f(e),PXmp2[02])) 
        //    Y[2] = exp(PXmp3[00]) + exp(PXmp3[01]*pow(f(e),PXmp3[02])) 
        //    Y[3] = exp(PXmp4[00]) + exp(PXmp4[01]*pow(f(e),PXmp4[02])) 
        // with 
        //    f(e)  = 0.5 + 20*(log(e)-log01)/(log10-log01)
        //    f'(e) = 20/(log10-log01)/e
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * PXmp1[01]*PXmp1[02]*f_e^(PXmp1[02]-1)*exp{}*f'(e) +
        //               dBdY[0][1] * PXmp2[01]*PXmp2[02]*f_e^(PXmp2[02]-1)*exp{}*f'(e) +
        //               dBdY[0][2] * PXmp3[01]*PXmp3[02]*f_e^(PXmp3[02]-1)*exp{}*f'(e) +   
        //               dBdY[0][3] * PXmp4[01]*PXmp4[02]*f_e^(PXmp4[02]-1)*exp{}*f'(e)    
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp0/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        // ----------------------------------------------------------------------------
        // For thisp1, the dependence of the four values is
        //    thisp1 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        //    Y[0] = PXmp1[10] + PXmp1[11] * f(e) + PXmp1[12] * f(e)^2 
        //    Y[1] = PXmp2[10] + PXmp2[11] * f(e) + PXmp2[12] * f(e)^2 
        //    Y[2] = PXmp3[10] + PXmp3[11] * f(e) + PXmp3[12] * f(e)^2 
        //    Y[3] = PXmp4[10] + PXmp4[11] * f(e) + PXmp4[12] * f(e)^2 
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * (PXmp1[11] * f'(e) + PXmp1[12] * 2*f(e)*f'(e)) +
        //               dBdY[0][1] * (PXmp2[11] * f'(e) + PXmp2[12] * 2*f(e)*f'(e)) +
        //               dBdY[0][2] * (PXmp3[11] * f'(e) + PXmp3[12] * 2*f(e)*f'(e)) +   
        //               dBdY[0][3] * (PXmp4[11] * f'(e) + PXmp4[12] * 2*f(e)*f'(e))    
        // and similarly
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp1/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        //
        // For thisp2, the dependence of the four values is
        //    thisp2 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        //    Y[0] = PXmp1[20] + PXmp1[21] * f(e) + PXmp1[22] * f(e)^2 
        //    Y[1] = PXmp2[20] + PXmp2[21] * f(e) + PXmp2[22] * f(e)^2 
        //    Y[2] = PXmp3[20] + PXmp3[21] * f(e) + PXmp3[22] * f(e)^2 
        //    Y[3] = PXmp4[20] + PXmp4[21] * f(e) + PXmp4[22] * f(e)^2 
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * (PXmp1[21] * f'(e) + PXmp1[22] * 2*f(e)*f'(e)) +
        //               dBdY[0][1] * (PXmp2[21] * f'(e) + PXmp2[22] * 2*f(e)*f'(e)) +
        //               dBdY[0][2] * (PXmp3[21] * f'(e) + PXmp3[22] * 2*f(e)*f'(e)) +   
        //               dBdY[0][3] * (PXmp4[21] * f'(e) + PXmp4[22] * 2*f(e)*f'(e))    
        // and similarly
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        //
        // This is the calculation of the four cubic parameters B[]
        // --------------------------------------------------------    
        computecubicpars (1); // This will compute B[] given Y[]

        // Now all ingredient for dthisp0/de, dthisp1/de, dthisp2/de are there
        // -------------------------------------------------------------------
        double f_e      = 0.5 + 20.*(log(energy)-log_01)/logdif;
        double fprime_e = 20./logdif/energy;
        double x        = 0.5 + 4. * theta/thetamax;
        double dB0de, dB1de, dB2de, dB3de;
        if (parnumber==0) {
        //    dthisp0/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * PXmp1[01]*(PXmp1[02]-1)*exp{}*f'(e) +
        //               dBdY[0][1] * PXmp2[01]*(PXmp2[02]-1)*exp{}*f'(e) +
        //               dBdY[0][2] * PXmp3[01]*(PXmp3[02]-1)*exp{}*f'(e) +   
        //               dBdY[0][3] * PXmp4[01]*(PXmp4[02]-1)*exp{}*f'(e)    
            double f0 = PXmp1_p[0][1]*PXmp1_p[0][2]*pow(f_e,PXmp1_p[0][2]-1.)*exp(PXmp1_p[0][1]*pow(f_e,PXmp1_p[0][2]))*fprime_e;
            double f1 = PXmp2_p[0][1]*PXmp2_p[0][2]*pow(f_e,PXmp2_p[0][2]-1.)*exp(PXmp2_p[0][1]*pow(f_e,PXmp2_p[0][2]))*fprime_e;
            double f2 = PXmp3_p[0][1]*PXmp3_p[0][2]*pow(f_e,PXmp3_p[0][2]-1.)*exp(PXmp3_p[0][1]*pow(f_e,PXmp3_p[0][2]))*fprime_e;
            double f3 = PXmp4_p[0][1]*PXmp4_p[0][2]*pow(f_e,PXmp4_p[0][2]-1.)*exp(PXmp4_p[0][1]*pow(f_e,PXmp4_p[0][2]))*fprime_e;
            dB0de = dBdY[0][0] * f0 +
                    dBdY[0][1] * f1 +
                    dBdY[0][2] * f2 +
                    dBdY[0][3] * f3;
            dB1de = dBdY[1][0] * f0 +
                    dBdY[1][1] * f1 +
                    dBdY[1][2] * f2 +
                    dBdY[1][3] * f3;
            dB2de = dBdY[2][0] * f0 +
                    dBdY[2][1] * f1 +
                    dBdY[2][2] * f2 +
                    dBdY[2][3] * f3;
            dB3de = dBdY[3][0] * f0 +
                    dBdY[3][1] * f1 +
                    dBdY[3][2] * f2 +
                    dBdY[3][3] * f3;
        } else if (parnumber==2) { // No parnumber 1 for muons
            //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * (PXmp1[21] * f'(e) + PXmp1[22] * 2*f(e)*f'(e)) +
            //               dBdY[0][1] * (PXmp2[21] * f'(e) + PXmp2[22] * 2*f(e)*f'(e)) +
            //               dBdY[0][2] * (PXmp3[21] * f'(e) + PXmp3[22] * 2*f(e)*f'(e)) +   
            //               dBdY[0][3] * (PXmp4[21] * f'(e) + PXmp4[22] * 2*f(e)*f'(e))    
            double f1 = PXmp1_p[2][1]*fprime_e + PXmp1_p[2][2]*2.*f_e*fprime_e;
            double f2 = PXmp2_p[2][1]*fprime_e + PXmp2_p[2][2]*2.*f_e*fprime_e;
            double f3 = PXmp3_p[2][1]*fprime_e + PXmp3_p[2][2]*2.*f_e*fprime_e;
            double f4 = PXmp4_p[2][1]*fprime_e + PXmp4_p[2][2]*2.*f_e*fprime_e;
            dB0de = dBdY[0][0] * f1 + 
                    dBdY[0][1] * f2 + 
                    dBdY[0][2] * f3 + 
                    dBdY[0][3] * f4;
            dB1de = dBdY[1][0] * f1 + 
                    dBdY[1][1] * f2 + 
                    dBdY[1][2] * f3 + 
                    dBdY[1][3] * f4;
            dB2de = dBdY[2][0] * f1 + 
                    dBdY[2][1] * f2 + 
                    dBdY[2][2] * f3 + 
                    dBdY[2][3] * f4;
            dB3de = dBdY[3][0] * f1 + 
                    dBdY[3][1] * f2 + 
                    dBdY[3][2] * f3 + 
                    dBdY[3][3] * f4;
        }
        return dB0de + dB1de*x + dB2de*x*x + dB3de*x*x*x;
    } else if (mode==3 || mode==32) { // Derivative wrt theta or d2/dtheta2
        computecubicpars (0);
        // Now B[] contains the four parameters of the cubic
        double x = 0.5 + 4. * theta/thetamax;
        double dxdtheta = 4./thetamax;
        if (mode==3) {
            return (B[1] + B[2]*2.*x + B[3] * 3.*x*x) * dxdtheta;
        } else if (mode==32) {
            return (2.*B[2] + 6.*B[3]*x)*dxdtheta; // No d2xdtheta2 term, null
        }
    } 
    // If it gets here an invalid value of mode was passed
    // ---------------------------------------------------
    cout    << "Warning - invalid mode for cubic in solvecubic_mp" << endl;
    outfile << "Warning - invalid mode for cubic in solvecubic_mp" << endl;
    warnings1++;
    return 0.; // This should not happen
}

// This function obtains the value of parameters thisp0, thisp1, thisp2 for e+g from protons, or derivatives
// ---------------------------------------------------------------------------------------------------------
double solvecubic_ep (int parnumber, double energy, double theta, int mode) {

    if (mode==0) { // Primal value
        computecubicpars (0); // This will compute B[] given Y[]
        // We proceed to compute the four cubic parameters 
        // Now B[] contains the four parameters of the cubic
        double val = 0.;
        double x = 0.5 + 4. * theta/thetamax;
        for (int i=0; i<4; i++) {
            val += B[i] * pow(x,i);
        }
        return val;
    } else if (mode==2) { // Get derivative wrt energy
        // In the above calculation, the value of the cubic at the four points
        // depends on energy through their parametrization, which is different
        // for the three parameters thisp0, thisp1, thisp2.
        //    thisp0 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        // For thisp0, the dependence of the four values is
        //    Y[0] = exp(PXep1[00]) + exp(PXep1[01]*pow(f(e),PXep1[02])) 
        //    Y[1] = exp(PXep2[00]) + exp(PXep2[01]*pow(f(e),PXep2[02])) 
        //    Y[2] = exp(PXep3[00]) + exp(PXep3[01]*pow(f(e),PXep3[02])) 
        //    Y[3] = exp(PXep4[00]) + exp(PXep4[01]*pow(f(e),PXep4[02])) 
        // with 
        //    f(e)  = 0.5 + 20*(log(e)-log01)/(log10-log01)
        //    f'(e) = 20/(log10-log01)/e
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * PXep1[01]*PXep1[02]*f_e^(PXep1[02]-1)*exp{}*f'(e) +
        //               dBdY[0][1] * PXep2[01]*PXep2[02]*f_e^(PXep2[02]-1)*exp{}*f'(e) +
        //               dBdY[0][2] * PXep3[01]*PXep3[02]*f_e^(PXep3[02]-1)*exp{}*f'(e) +   
        //               dBdY[0][3] * PXep4[01]*PXep4[02]*f_e^(PXep4[02]-1)*exp{}*f'(e)    
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp0/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        // ----------------------------------------------------------------------------
        // For thisp1, the dependence of the four values is
        //    thisp1 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        //    Y[0] = PXep1[10] + PXep1[11] * f(e) + PXep1[12] * f(e)^2 
        //    Y[1] = PXep2[10] + PXep2[11] * f(e) + PXep2[12] * f(e)^2 
        //    Y[2] = PXep3[10] + PXep3[11] * f(e) + PXep3[12] * f(e)^2 
        //    Y[3] = PXep4[10] + PXep4[11] * f(e) + PXep4[12] * f(e)^2 
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * (PXep1[11] * f'(e) + PXep1[12] * 2*f(e)*f'(e)) +
        //               dBdY[0][1] * (PXep2[11] * f'(e) + PXep2[12] * 2*f(e)*f'(e)) +
        //               dBdY[0][2] * (PXep3[11] * f'(e) + PXep3[12] * 2*f(e)*f'(e)) +   
        //               dBdY[0][3] * (PXep4[11] * f'(e) + PXep4[12] * 2*f(e)*f'(e))    
        // and similarly
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp1/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        // 
        // For thisp2, the dependence of the four values is
        //    thisp2 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        //    Y[0] = PXep1[20] + PXep1[21] * f(e) + PXep1[22] * f(e)^2 
        //    Y[1] = PXep2[20] + PXep2[21] * f(e) + PXep2[22] * f(e)^2 
        //    Y[2] = PXep3[20] + PXep3[21] * f(e) + PXep3[22] * f(e)^2 
        //    Y[3] = PXep4[20] + PXep4[21] * f(e) + PXep4[22] * f(e)^2 
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * (PXep1[21] * f'(e) + PXep1[22] * 2*f(e)*f'(e)) +
        //               dBdY[0][1] * (PXep2[21] * f'(e) + PXep2[22] * 2*f(e)*f'(e)) +
        //               dBdY[0][2] * (PXep3[21] * f'(e) + PXep3[22] * 2*f(e)*f'(e)) +   
        //               dBdY[0][3] * (PXep4[21] * f'(e) + PXep4[22] * 2*f(e)*f'(e))    
        // and similarly
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        //
        // This is the calculation of the four cubic parameters B[]
        // --------------------------------------------------------
        computecubicpars (1); // <-- this will compute B[] given Y[]

        // Now all ingredient for dthisp0/de, dthisp1/de, dthisp2/de are there
        // -------------------------------------------------------------------
        double f_e      = 0.5 + 20.*(log(energy)-log_01)/logdif;
        double fprime_e = 20./logdif/energy;
        double x        = 0.5 + 4. * theta/thetamax;
        double dB0de, dB1de, dB2de, dB3de;
        if (parnumber==0) {
            //    dthisp0/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * PXep1[01]*(PXep1[02]-1)*exp{}*f'(e) +
            //               dBdY[0][1] * PXep2[01]*(PXep2[02]-1)*exp{}*f'(e) +
            //               dBdY[0][2] * PXep3[01]*(PXep3[02]-1)*exp{}*f'(e) +   
            //               dBdY[0][3] * PXep4[01]*(PXep4[02]-1)*exp{}*f'(e)    
            double f0 = PXep1_p[0][1]*PXep1_p[0][2]*pow(f_e,PXep1_p[0][2]-1.)*exp(PXep1_p[0][1]*pow(f_e,PXep1_p[0][2]))*fprime_e;
            double f1 = PXep2_p[0][1]*PXep2_p[0][2]*pow(f_e,PXep2_p[0][2]-1.)*exp(PXep2_p[0][1]*pow(f_e,PXep2_p[0][2]))*fprime_e;
            double f2 = PXep3_p[0][1]*PXep3_p[0][2]*pow(f_e,PXep3_p[0][2]-1.)*exp(PXep3_p[0][1]*pow(f_e,PXep3_p[0][2]))*fprime_e;
            double f3 = PXep4_p[0][1]*PXep4_p[0][2]*pow(f_e,PXep4_p[0][2]-1.)*exp(PXep4_p[0][1]*pow(f_e,PXep4_p[0][2]))*fprime_e;
            dB0de = dBdY[0][0] * f0 +
                    dBdY[0][1] * f1 +
                    dBdY[0][2] * f2 +
                    dBdY[0][3] * f3;
            dB1de = dBdY[1][0] * f0 +
                    dBdY[1][1] * f1 +
                    dBdY[1][2] * f2 +
                    dBdY[1][3] * f3;
            dB2de = dBdY[2][0] * f0 +
                    dBdY[2][1] * f1 +
                    dBdY[2][2] * f2 +
                    dBdY[2][3] * f3;
            dB3de = dBdY[3][0] * f0 +
                    dBdY[3][1] * f1 +
                    dBdY[3][2] * f2 +
                    dBdY[3][3] * f3;
        } else if (parnumber==1) {  
            //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * (PXep1[11] * f'(e) + PXep1[12] * 2*f(e)*f'(e)) +
            //               dBdY[0][1] * (PXep2[11] * f'(e) + PXep2[12] * 2*f(e)*f'(e)) +
            //               dBdY[0][2] * (PXep3[11] * f'(e) + PXep3[12] * 2*f(e)*f'(e)) +   
            //               dBdY[0][3] * (PXep4[11] * f'(e) + PXep4[12] * 2*f(e)*f'(e))    
            double f1 = PXep1_p[1][1]*fprime_e + PXep1_p[1][2]*2.*f_e*fprime_e;
            double f2 = PXep2_p[1][1]*fprime_e + PXep2_p[1][2]*2.*f_e*fprime_e;
            double f3 = PXep3_p[1][1]*fprime_e + PXep3_p[1][2]*2.*f_e*fprime_e;
            double f4 = PXep4_p[1][1]*fprime_e + PXep4_p[1][2]*2.*f_e*fprime_e;
            dB0de = dBdY[0][0] * f1 + 
                    dBdY[0][1] * f2 + 
                    dBdY[0][2] * f3 + 
                    dBdY[0][3] * f4;
            dB1de = dBdY[1][0] * f1 + 
                    dBdY[1][1] * f2 + 
                    dBdY[1][2] * f3 + 
                    dBdY[1][3] * f4;
            dB2de = dBdY[2][0] * f1 + 
                    dBdY[2][1] * f2 + 
                    dBdY[2][2] * f3 + 
                    dBdY[2][3] * f4;
            dB3de = dBdY[3][0] * f1 + 
                    dBdY[3][1] * f2 + 
                    dBdY[3][2] * f3 + 
                    dBdY[3][3] * f4;
        } else if (parnumber==2) { 
            //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * (PXep1[21] * f'(e) + PXep1[22] * 2*f(e)*f'(e)) +
            //               dBdY[0][1] * (PXep2[21] * f'(e) + PXep2[22] * 2*f(e)*f'(e)) +
            //               dBdY[0][2] * (PXep3[21] * f'(e) + PXep3[22] * 2*f(e)*f'(e)) +   
            //               dBdY[0][3] * (PXep4[21] * f'(e) + PXep4[22] * 2*f(e)*f'(e))    
            double f1 = PXep1_p[2][1]*fprime_e + PXep1_p[2][2]*2.*f_e*fprime_e;
            double f2 = PXep2_p[2][1]*fprime_e + PXep2_p[2][2]*2.*f_e*fprime_e;
            double f3 = PXep3_p[2][1]*fprime_e + PXep3_p[2][2]*2.*f_e*fprime_e;
            double f4 = PXep4_p[2][1]*fprime_e + PXep4_p[2][2]*2.*f_e*fprime_e;
            dB0de = dBdY[0][0] * f1 + 
                    dBdY[0][1] * f2 + 
                    dBdY[0][2] * f3 + 
                    dBdY[0][3] * f4;
            dB1de = dBdY[1][0] * f1 + 
                    dBdY[1][1] * f2 + 
                    dBdY[1][2] * f3 + 
                    dBdY[1][3] * f4;
            dB2de = dBdY[2][0] * f1 + 
                    dBdY[2][1] * f2 + 
                    dBdY[2][2] * f3 + 
                    dBdY[2][3] * f4;
            dB3de = dBdY[3][0] * f1 + 
                    dBdY[3][1] * f2 + 
                    dBdY[3][2] * f3 + 
                    dBdY[3][3] * f4;
        }
        return dB0de + dB1de*x + dB2de*x*x + dB3de*x*x*x;
    } else if (mode==3 || mode==32) { // Derivative wrt theta
        computecubicpars(0);
        // Now B[] contains the four parameters of the cubic
        double x = 0.5 + 4. * theta/thetamax;
        double dxdtheta = 4./thetamax;
        if (mode==3) {
            return (B[1] + B[2]*2.*x + B[3] * 3.*x*x) * dxdtheta;
        } else if (mode==32) {
            return (2.*B[2] + 6.*B[3]*x)*dxdtheta; // No d2xdtheta2 term, null
        }
    } 
    // If it gets here an invalid value of mode was passed
    // ---------------------------------------------------
    cout    << "Warning - invalid mode for cubic in solvecubic_ep" << endl;
    outfile << "Warning - invalid mode for cubic in solvecubic_ep" << endl;
    warnings1++;
    return 0.; // This should not happen
}

// Function that corrects for the finite tank size when computing fluxes: the flux
// is evaluated at the center of the tank, but the tank has a finite size and this
// modifies the calculation if the distance (tank center, core) is small. 
// See graphs of Ratio_EG, Ratio_MG, ... filled up later, used to derive the
// correction function, which for now is a simple one:
// f(R) = 1 + a(x+1)exp(-x)/(x^2+b)
// -------------------------------------------------------------------------------
double FluxCorr (double R, int part, int mode=0) {
    double a, b;
    if (part==0)        { // EG
        a = -0.25127;
        b = 0.26997;
    } else if (part==1) { // EP
        a = -0.2612;
        b = 0.29185;
    } else if (part==2) { // MG
        a = -0.14106;
        b = 0.27065;
    } else if (part==3) { // MP
        a = -0.14111;
        b = 0.27054;
    }
    if (mode==0) {
        return 1.+a*(R+1.)*exp(-R)/(R*R+b);
    } else if (mode==1) {
        double dfdr = -R*exp(-R)*(a*R*R+2.*a*R+2.*a+a*b)/pow(R*R+b,2.);
        return dfdr;
    } else if (mode==2) {
        double dfdr = -R*exp(-R)*(a*R*R+2.*a*R+2.*a+a*b)/pow(R*R+b,2.); 
        return dfdr*(1./R - 1. +2*R/(R*R+2*a+2.+b) -4*R/(R*R+b));
    }
    cout << "Warning this should not happen" << endl;
    return 0.;
}


// Function parametrizing muon content in gamma showers
// ----------------------------------------------------
double MFromG (double energy, double theta, double R, int mode, double dRdTh=0, double d2RdTh2=0) {

    // Protect against out of range values
    // -----------------------------------
    if (R<Rmin) R = Rmin; 
    if (energy<0.1 || energy>10.) return 0.;
    if (theta<0. || theta>thetamax) return 0.;

    // Convert energy into the function we use in the interpolation
    // ------------------------------------------------------------
    double xe = 0.5+20.*(log(energy)-log_01)/logdif; // Energy is in PeV
    double xe2 = xe*xe;

    // Interpolate the two parameters given the wanted energy and theta
    // ----------------------------------------------------------------
    double thisp0, thisp2;

    int ielow = (int)(energy/0.1)-1;
    int iehig = ielow + 1;
    double de = (energy-ielow*0.1-0.1)/0.1;
    int itlow = (int)(theta*99./thetamax);
    int ithig = itlow + 1;
    double dt = (theta - itlow*(thetamax/99.))/(thetamax/99.);
    // Handle boundaries first
    if (iehig==100) {
        if (ithig==100) { // Upper edge of grid
            thisp0 = thisp0_mg[99][99];
            thisp2 = thisp2_mg[99][99];
        } else { // Move on energy edge interpolating in theta
            thisp0 = thisp0_mg[99][itlow] + dt*(thisp0_mg[99][ithig]-thisp0_mg[99][itlow]);
            thisp2 = thisp2_mg[99][itlow] + dt*(thisp2_mg[99][ithig]-thisp2_mg[99][itlow]);
        }
    } else if (ithig==100) { // Theta edge, interpolate in energy
        thisp0 = thisp0_mg[ielow][99] + de*(thisp0_mg[iehig][99]-thisp0_mg[ielow][99]);
        thisp2 = thisp2_mg[ielow][99] + de*(thisp2_mg[iehig][99]-thisp2_mg[ielow][99]);
    } else {
        // Do 2D interpolation
        double lowe0 = thisp0_mg[ielow][itlow] + dt*(thisp0_mg[ielow][ithig]-thisp0_mg[ielow][itlow]);
        double hige0 = thisp0_mg[iehig][itlow] + dt*(thisp0_mg[iehig][ithig]-thisp0_mg[iehig][itlow]);
        thisp0       = lowe0 + de*(hige0-lowe0);
        double lowe2 = thisp2_mg[ielow][itlow] + dt*(thisp2_mg[ielow][ithig]-thisp2_mg[ielow][itlow]);
        double hige2 = thisp2_mg[iehig][itlow] + dt*(thisp2_mg[iehig][ithig]-thisp2_mg[iehig][itlow]);
        thisp2       = lowe2 + de*(hige2-lowe2);
    }

    double flux0 = TankArea*0.02*thisp0*exp(-1.*pow(R,thisp2)); // Function for muons. Note factor of 50
    double flux  = flux0 + fluxB_mu;
    // Compute correction factor accounting for physical size of tank
    // -------------------------------------------------------------- 
    // double FC = FluxCorr(R,2,0);

    // Note: there is now (12/4/24, v99 onwards) background introduced, which produces the need to model the fraction of signal 
    // as a function of parameters thisp0, thisp2.
    // The flow is the following: we have a background component in the flux now,
    //   flux = flux0 + fluxB
    // which does not affect the derivatives we already computed:
    //   dflux/dx = dflux0/dx and all the other derivatives are untouched, as the B term is constant.
    // However we now have:
    //   f_b = fluxB/(flux0+fluxB)
    //   df_b/dx = -f_b/(flux0+fluxB)*dflux0/dx
    // which we can extract from the original derivatives of the flux, computed for mode!=0 below.  
    // The above has an impact in the timing distribution and the derivatives, used e.g. in shower reconstruction likelihood:
    //   P(tmu) = G(tmu,sigma)*(1-f_b) + f_b/T_range
    // where T_range is the time range we want to give to background muon counts. The derivative of the pdf is now
    //   dP(tmu)/dx = dG(tmu,sigma)/dx*(1-f_b) - G(tmu,sigma) * df_b/dx
    // -----------------------------------------------------------------------------------------------------------------------

    if (mode==0) { // Return function value
        // flux0 = flux0*FC; // correct for tank size
        if (flux0>largenumber) return largenumber;
        if (flux0<epsilon2) return epsilon2;
        if (flux0!=flux0) {
            cout    << "Warning flux mg; E,T,R = " << energy << " " << theta << " " << R << endl;
            outfile << "Warning flux mg; E,T,R = " << energy << " " << theta << " " << R << endl;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.; // Protect against nans
        }
        return flux0;
    } else if (mode==1) { // Return derivative with respect to R
        double dfluxdR = -flux0 * thisp2*pow(R,thisp2-1.);  
        if (dfluxdR!=dfluxdR) {
            cout    << "Warning dfluxdR mg" << endl; 
            outfile << "Warning dfluxdR mg" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.;
        }
        // Add derivative of correction factor
        // -----------------------------------
        // dfluxdR = dfluxdR * FC + flux0*FluxCorr(R,2,1);
        return dfluxdR; 
    } else if (mode==12) { // Return second derivative wrt R
        double d2fluxdR2 = flux0 * (thisp2*pow(R,2.*thisp2-2.)-thisp2*pow(R,thisp2-2));
        if (d2fluxdR2!=d2fluxdR2) {
            cout    << "Warning d2fluxdR2 mg" << endl; 
            outfile << "Warning d2fluxdR2 mg" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.;
        }
        // Add derivative of correction factor
        // -----------------------------------
        // double dfluxdR = -flux0 * thisp2*pow(R,thisp2-1.);  
        // d2fluxdR2 = d2fluxdR2 * FC + 2.*dfluxdR*FluxCorr(R,2,1) + flux0*FluxCorr(R,2,2);
        return d2fluxdR2; 
    } else if (mode==2) { // Return derivative with respect to energy or d2/de2
        // Note, this is tricky: we are deriving 
        // f = p0(e)*exp(-R^p2(e)) over de.
        // This is like dh(g(f(e)))/de = h'(g(f))g'(f)f'(e),
        // So we get A*exp()*f(g(e))*f'(g(e))*g'(e)
        // In our case this is dp0/de * exp() + p0(e) * d(exp)/de =
        //                     dp0/de * f/p0 + f * [-R^(p2)] log(R) dp2/de 
        // Also note: 
        // we are neglecting the variation of the flux on E due to a variation of R on E.
        // That is, we take R constant in the calculation, ignoring the term flux*(-p2*R^{p2-1})*dR/dE
        // This seems ok for the application we have (computing dE/dR by fixing small increments of 
        // E and R and determining when the dlogL/dE remains constant, see routine dE_dR)
        // -------------------------------------------------------------------------------------------

        // Interpolate dthisp0de, dthisp2de - Handle boundaries first
        double dthisp0de, dthisp2de;
        if (iehig==100) {
            if (ithig==100) { // Upper edge of grid
                dthisp0de = dthisp0de_mg[99][99];
                dthisp2de = dthisp2de_mg[99][99];
            } else { // Move on energy edge interpolating in theta
                dthisp0de = dthisp0de_mg[99][itlow] + dt*(dthisp0de_mg[99][ithig]-dthisp0de_mg[99][itlow]);
                dthisp2de = dthisp2de_mg[99][itlow] + dt*(dthisp2de_mg[99][ithig]-dthisp2de_mg[99][itlow]);
            }
        } else if (ithig==100) { // Theta edge, interpolate in energy
            dthisp0de = dthisp0de_mg[ielow][99] + de*(dthisp0de_mg[iehig][99]-dthisp0de_mg[ielow][99]);
            dthisp2de = dthisp2de_mg[ielow][99] + de*(dthisp2de_mg[iehig][99]-dthisp2de_mg[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0de_mg[ielow][itlow] + dt*(dthisp0de_mg[ielow][ithig]-dthisp0de_mg[ielow][itlow]);
            double hige0 = dthisp0de_mg[iehig][itlow] + dt*(dthisp0de_mg[iehig][ithig]-dthisp0de_mg[iehig][itlow]);
            dthisp0de    = lowe0 + de*(hige0-lowe0);
            double lowe2 = dthisp2de_mg[ielow][itlow] + dt*(dthisp2de_mg[ielow][ithig]-dthisp2de_mg[ielow][itlow]);
            double hige2 = dthisp2de_mg[iehig][itlow] + dt*(dthisp2de_mg[iehig][ithig]-dthisp2de_mg[iehig][itlow]);
            dthisp2de    = lowe2 + de*(hige2-lowe2);
        }

        double dfluxde = flux0*(1./thisp0*dthisp0de -pow(R,thisp2)*log(R)*dthisp2de);
        if (dfluxde!=dfluxde) {
            cout    << "Warning dfluxde mg; R = " << R << " thisp0, p2 = " << thisp0 << " " << thisp2 << " " << dthisp0de << " " << dthisp2de << endl; 
            outfile << "Warning dfluxde mg; R = " << R << " thisp0, p2 = " << thisp0 << " " << thisp2 << " " << dthisp0de << " " << dthisp2de << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.;
        }
        // Consider flux correction
        // ------------------------
        // return dfluxde*FC;
        return dfluxde;
    } else if (mode==22 || mode==23 || mode==24 || mode==25) { // Return other derivatives wrt energy and distance
        // 2 = dflux/de
        // 22= d2flux/de2
        // 23= d2flux/dedr
        // 24= d3flux/de2dr
        // 25= d3flux/de3
        // Interpolate dthisp0de, dthisp2de - Handle boundaries first
        double dthisp0de, dthisp2de;
        double d2thisp0de2, d2thisp2de2;
        double d3thisp0de3, d3thisp2de3;
        if (iehig==100) {
            if (ithig==100) { // Upper edge of grid
                dthisp0de = dthisp0de_mg[99][99];
                dthisp2de = dthisp2de_mg[99][99];
                d2thisp0de2 = d2thisp0de2_mg[99][99];
                d2thisp2de2 = d2thisp2de2_mg[99][99];
                d3thisp0de3 = d3thisp0de3_mg[99][99];
                d3thisp2de3 = d3thisp2de3_mg[99][99];
            } else { // Move on energy edge interpolating in theta
                dthisp0de   = dthisp0de_mg[99][itlow]   + dt*(dthisp0de_mg[99][ithig]-dthisp0de_mg[99][itlow]);
                dthisp2de   = dthisp2de_mg[99][itlow]   + dt*(dthisp2de_mg[99][ithig]-dthisp2de_mg[99][itlow]);
                d2thisp0de2 = d2thisp0de2_mg[99][itlow] + dt*(d2thisp0de2_mg[99][ithig]-d2thisp0de2_mg[99][itlow]);
                d2thisp2de2 = d2thisp2de2_mg[99][itlow] + dt*(d2thisp2de2_mg[99][ithig]-d2thisp2de2_mg[99][itlow]);
                d3thisp0de3 = d3thisp0de3_mg[99][itlow] + dt*(d3thisp0de3_mg[99][ithig]-d3thisp0de3_mg[99][itlow]);
                d3thisp2de3 = d3thisp2de3_mg[99][itlow] + dt*(d3thisp2de3_mg[99][ithig]-d3thisp2de3_mg[99][itlow]);
            }
        } else if (ithig==100) { // Theta edge, interpolate in energy
            dthisp0de   = dthisp0de_mg[ielow][99]   + de*(dthisp0de_mg[iehig][99]-dthisp0de_mg[ielow][99]);
            dthisp2de   = dthisp2de_mg[ielow][99]   + de*(dthisp2de_mg[iehig][99]-dthisp2de_mg[ielow][99]);
            d2thisp0de2 = d2thisp0de2_mg[ielow][99] + de*(d2thisp0de2_mg[iehig][99]-d2thisp0de2_mg[ielow][99]);
            d2thisp2de2 = d2thisp2de2_mg[ielow][99] + de*(d2thisp2de2_mg[iehig][99]-d2thisp2de2_mg[ielow][99]);
            d3thisp0de3 = d3thisp0de3_mg[ielow][99] + de*(d3thisp0de3_mg[iehig][99]-d3thisp0de3_mg[ielow][99]);
            d3thisp2de3 = d3thisp2de3_mg[ielow][99] + de*(d3thisp2de3_mg[iehig][99]-d3thisp2de3_mg[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0de_mg[ielow][itlow] + dt*(dthisp0de_mg[ielow][ithig]-dthisp0de_mg[ielow][itlow]);
            double hige0 = dthisp0de_mg[iehig][itlow] + dt*(dthisp0de_mg[iehig][ithig]-dthisp0de_mg[iehig][itlow]);
            dthisp0de    = lowe0 + de*(hige0-lowe0);
            double lowe2 = dthisp2de_mg[ielow][itlow] + dt*(dthisp2de_mg[ielow][ithig]-dthisp2de_mg[ielow][itlow]);
            double hige2 = dthisp2de_mg[iehig][itlow] + dt*(dthisp2de_mg[iehig][ithig]-dthisp2de_mg[iehig][itlow]);
            dthisp2de    = lowe2 + de*(hige2-lowe2);
            // Second derivative
            lowe0 = d2thisp0de2_mg[ielow][itlow] + dt*(d2thisp0de2_mg[ielow][ithig]-d2thisp0de2_mg[ielow][itlow]);
            hige0 = d2thisp0de2_mg[iehig][itlow] + dt*(d2thisp0de2_mg[iehig][ithig]-d2thisp0de2_mg[iehig][itlow]);
            d2thisp0de2  = lowe0 + de*(hige0-lowe0);
            lowe2 = d2thisp2de2_mg[ielow][itlow] + dt*(d2thisp2de2_mg[ielow][ithig]-d2thisp2de2_mg[ielow][itlow]);
            hige2 = d2thisp2de2_mg[iehig][itlow] + dt*(d2thisp2de2_mg[iehig][ithig]-d2thisp2de2_mg[iehig][itlow]);
            d2thisp2de2  = lowe2 + de*(hige2-lowe2);
            // Third derivative
            lowe0 = d3thisp0de3_mg[ielow][itlow] + dt*(d3thisp0de3_mg[ielow][ithig]-d3thisp0de3_mg[ielow][itlow]);
            hige0 = d3thisp0de3_mg[iehig][itlow] + dt*(d3thisp0de3_mg[iehig][ithig]-d3thisp0de3_mg[iehig][itlow]);
            d3thisp0de3    = lowe0 + de*(hige0-lowe0);
            lowe2 = d3thisp2de3_mg[ielow][itlow] + dt*(d3thisp2de3_mg[ielow][ithig]-d3thisp2de3_mg[ielow][itlow]);
            hige2 = d3thisp2de3_mg[iehig][itlow] + dt*(d3thisp2de3_mg[iehig][ithig]-d3thisp2de3_mg[iehig][itlow]);
            d3thisp2de3    = lowe2 + de*(hige2-lowe2);
        }
        //
        if (mode==22) { // 22= d2flux/de2
            double logR       = log(R);
            double Rtop2      = pow(R,thisp2);
            //double Cfactor    = dthisp0de/thisp0 - Rtop2*logR*dthisp2de;
            //double dCfactorde = d2thisp0de2/thisp0 -pow(dthisp0de/thisp0,2.) -pow(Rtop2*logR*dthisp2de,2.) - Rtop2*logR*d2thisp2de2; 
            //double d2fluxde2_old  = flux0 * (Cfactor*Cfactor+dCfactorde);
            //
            //                   pow(1./thisp0*dthisp0de -Rtop2*logR*dthisp2de,2.)  
            //                   + pow(dthisp0de/thisp0,2.) + d2thisp0de2/thisp0 
            //                   - Rtop2*pow(logR*dthisp2de,2.)
            //                   - Rtop2*logR*d2thisp2de2 
            //                );
            double d2fluxde2 = flux0*(-2./thisp0*Rtop2*logR*dthisp0de*dthisp2de
                               -Rtop2*pow(logR*dthisp2de,2.)
                               +pow(Rtop2*logR*dthisp2de,2.)
                               +d2thisp0de2/thisp0-Rtop2*logR*d2thisp2de2);
            //if (d2fluxde2_math!=0) cout << d2fluxde2/d2fluxde2_math << endl;
            // From Mathematica:
            // d2f0de2 = -2 f0/p0 R^p2 logR p0' p2' 
            //           - f0 R^p2 logR^2 p2'^2
            //           + f0 R^2p2 logR^2 p2'^2
            //           + f0 p0''/p0 - f0 R^p2 logR p2'' =
            //           f0 [ -2/p0 R^p2 logR p0' p2' 
            //                -R^p2 logR^2 p2'^2 
            //                +R^2p2 logR^2 p2'^2
            //                +p0''/p0 - R^p2 logR p2''] 
            // ---------------------------------------------------------------
            if (d2fluxde2!=d2fluxde2) {
                cout    << "Warning d2fluxde2 mg; R = " << R << " thisp0, p2 = " << thisp0 << " " << thisp2 << " " << dthisp0de << " " << dthisp2de << endl; 
                outfile << "Warning d2fluxde2 mg; R = " << R << " thisp0, p2 = " << thisp0 << " " << thisp2 << " " << dthisp0de << " " << dthisp2de << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
                return 0.;
            }
            // return d2fluxde2*FC; // include flux correction for tank size
            return d2fluxde2;
        } else if (mode==23) { // Derivative of d2flux wrt de dr
            double Rtop2   = pow(R,thisp2);
            double dfluxdR = -flux0 * thisp2*pow(R,thisp2-1.);  
            double logR = log(R);
            // double d2fluxdedr = dfluxdR * (1./thisp0*dthisp0de -Rtop2*logR*dthisp2de)
            //                    - flux0 * pow(R,thisp2-1)*dthisp2de*(thisp2*logR+1.); 
            double d2fluxdedr = flux0 * ( - pow(R,thisp2-1.)*dthisp0de*thisp2/thisp0 
                                          - pow(R,thisp2-1.)*dthisp2de     
                                          - pow(R,thisp2-1.)*logR*thisp2*dthisp2de
                                          + pow(R,2*thisp2-1.)*logR*thisp2*dthisp2de); 
            //if (dfluxdedr_mathem!=0) cout << d2fluxdedr/dfluxdedr_mathem << endl;
            if (d2fluxdedr!=d2fluxdedr) {
                cout    << "Warning d2fluxdedr mg; R = " << R << " thisp0, p2 = " << thisp0 << " " << thisp2 << " " << dthisp0de << " " << dthisp2de << endl; 
                outfile << "Warning d2fluxdedr mg; R = " << R << " thisp0, p2 = " << thisp0 << " " << thisp2 << " " << dthisp0de << " " << dthisp2de << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
                return 0.;
            }
            // For the corrected flux we need to take the df/de and consider also the derivative of fc over r
            // ----------------------------------------------------------------------------------------------
            // double dfluxde = flux0*(1./thisp0*dthisp0de -pow(R,thisp2)*log(R)*dthisp2de); // derivative before FC 
            // d2fluxdedr = d2fluxdedr*FC + dfluxde*FluxCorr(R,2,1);
            return d2fluxdedr;
        } else if (mode==24) { // d3flux / dr de2
            double dfluxdR = -flux0 * thisp2*pow(R,thisp2-1.);  
            double Rtop2   = pow(R,thisp2);
            double Rtop2m1 = pow(R,thisp2-1);
            double logR    = log(R);
            // Mathematica solution, see "derivatives of fluxes.nb", eq. 470
            // -------------------------------------------------------------
            double d3fluxdrde2 = flux0/thisp0*Rtop2m1*(2.*(-1.+(Rtop2-1.)*logR*thisp2)*dthisp0de*dthisp2de 
                                                           -thisp2*d2thisp0de2 
                                                           +thisp0*(-logR*(2.-2.*Rtop2+(1.-3.*Rtop2+pow(Rtop2,2.))*logR*thisp2)*pow(dthisp2de,2.)
                                                           +(-1.+(-1.+Rtop2)*logR*thisp2)*d2thisp2de2));
            if (d3fluxdrde2!=d3fluxdrde2) {
                cout    << "Warning d3fluxdrde2 mg; R = " << R << " thisp0, p2 = " << thisp0 << " " << thisp2 << " " << dthisp0de << " " << dthisp2de << endl; 
                outfile << "Warning d3fluxdrde2 mg; R = " << R << " thisp0, p2 = " << thisp0 << " " << thisp2 << " " << dthisp0de << " " << dthisp2de << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
                return 0.;
            }
            // To compute the corrected flux we need to call in the derivative of the uncorrected one:
            // d^3f'/de^2dr = d^3f/de^2dr * fc + d^2f/de^2 * dfc/dr
            // --------------------------------------------------------------------------------------- 
            // double d2fluxde2 = flux0*(-2./thisp0*Rtop2*logR*dthisp0de*dthisp2de
            //                    -Rtop2*pow(logR*dthisp2de,2.)
            //                    +pow(Rtop2*logR*dthisp2de,2.)
            //                    +d2thisp0de2/thisp0-Rtop2*logR*d2thisp2de2);
            // d3fluxdrde2 = d3fluxdrde2*FC + d2fluxde2*FluxCorr(R,2,1);
            return d3fluxdrde2;
        } else if (mode==25) { // 25= d3flux/de3
            double lr         = log(R);
            double lr2        = lr*lr;
            double lr3        = lr2*lr;
            double Rtop2      = pow(R,thisp2);
            double er2        = pow(ee,-Rtop2);
            // Mathematica solution, see "d3f0mde3.nb"
            // ---------------------------------------
            double d3fluxde3  = - 3.*er2*lr2*Rtop2*dthisp0de*pow(dthisp2de,2.) + 3.*er2*lr2*pow(Rtop2*dthisp2de,2.)*dthisp0de
                                - er2*lr3*Rtop2*thisp0*pow(dthisp2de,3.) + 3.*er2*lr3*pow(Rtop2*dthisp2de,3.)*thisp0/Rtop2
                                - er2*lr3*pow(Rtop2*dthisp2de,3.)*thisp0 - 3.*er2*lr*Rtop2*dthisp2de*d2thisp0de2
                                - 3.*er2*lr*Rtop2*dthisp0de*d2thisp2de2 - 3.*er2*lr2*Rtop2*thisp0*dthisp2de*d2thisp2de2
                                + 3.*er2*lr2*pow(Rtop2,2.)*thisp0*dthisp2de*d2thisp2de2 + er2*d3thisp0de3 - er2*lr*Rtop2*thisp0*d3thisp2de3;
            if (d3fluxde3!=d3fluxde3) {
                cout    << "Warning d3fluxde3 mg; R = " << R << " thisp0, p2 = " << thisp0 << " " << thisp2 << " " << dthisp0de << " " << dthisp2de << endl; 
                outfile << "Warning d3fluxde3 mg; R = " << R << " thisp0, p2 = " << thisp0 << " " << thisp2 << " " << dthisp0de << " " << dthisp2de << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
                return 0.;
            }
            // return d3fluxde3*FC; // account for finite size of tank
            return d3fluxde3;
        }  
    } else if (mode==3 || mode==31 || mode==32) { // return derivative with respect to theta
        // mode = 3 dflux/dtheta; mode=31 d^2flux/dthetadR; mode=32 d^2 flux/dtheta2

        // Interpolate dthisp0dth, dthisp2dth - Handle boundaries first
        double dthisp0dth, dthisp2dth;
        double d2thisp0dth2, d2thisp2dth2;
        if (iehig==100) {
            if (ithig==100) { // Upper edge of grid
                dthisp0dth = dthisp0dth_mg[99][99];
                dthisp2dth = dthisp2dth_mg[99][99];
                d2thisp0dth2 = dthisp0dth_mg[99][99];
                d2thisp2dth2 = dthisp2dth_mg[99][99];
            } else { // Move on energy edge interpolating in theta
                dthisp0dth = dthisp0dth_mg[99][itlow] + dt*(dthisp0dth_mg[99][ithig]-dthisp0dth_mg[99][itlow]);
                dthisp2dth = dthisp2dth_mg[99][itlow] + dt*(dthisp2dth_mg[99][ithig]-dthisp2dth_mg[99][itlow]);
                d2thisp0dth2 = d2thisp0dth2_mg[99][itlow]+dt*(d2thisp0dth2_mg[99][ithig]-d2thisp0dth2_mg[99][itlow]);
                d2thisp2dth2 = d2thisp2dth2_mg[99][itlow]+dt*(d2thisp2dth2_mg[99][ithig]-d2thisp2dth2_mg[99][itlow]);
            }
        } else if (ithig==100) { // Theta edge, interpolate in energy
            dthisp0dth = dthisp0dth_mg[ielow][99] + de*(dthisp0dth_mg[iehig][99]-dthisp0dth_mg[ielow][99]);
            dthisp2dth = dthisp2dth_mg[ielow][99] + de*(dthisp2dth_mg[iehig][99]-dthisp2dth_mg[ielow][99]);
            d2thisp0dth2 = d2thisp0dth2_mg[ielow][99] + de*(d2thisp0dth2_mg[iehig][99]-d2thisp0dth2_mg[ielow][99]);
            d2thisp2dth2 = d2thisp2dth2_mg[ielow][99] + de*(d2thisp2dth2_mg[iehig][99]-d2thisp2dth2_mg[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0dth_mg[ielow][itlow] + dt*(dthisp0dth_mg[ielow][ithig]-dthisp0dth_mg[ielow][itlow]);
            double hige0 = dthisp0dth_mg[iehig][itlow] + dt*(dthisp0dth_mg[iehig][ithig]-dthisp0dth_mg[iehig][itlow]);
            dthisp0dth   = lowe0 + de*(hige0-lowe0);
            double lowe2 = dthisp2dth_mg[ielow][itlow] + dt*(dthisp2dth_mg[ielow][ithig]-dthisp2dth_mg[ielow][itlow]);
            double hige2 = dthisp2dth_mg[iehig][itlow] + dt*(dthisp2dth_mg[iehig][ithig]-dthisp2dth_mg[iehig][itlow]);
            dthisp2dth   = lowe2 + de*(hige2-lowe2);
            lowe0 = d2thisp0dth2_mg[ielow][itlow] + dt*(d2thisp0dth2_mg[ielow][ithig]-d2thisp0dth2_mg[ielow][itlow]);
            hige0 = d2thisp0dth2_mg[iehig][itlow] + dt*(d2thisp0dth2_mg[iehig][ithig]-d2thisp0dth2_mg[iehig][itlow]);
            d2thisp0dth2   = lowe0 + de*(hige0-lowe0);
            lowe2 = d2thisp2dth2_mg[ielow][itlow] + dt*(d2thisp2dth2_mg[ielow][ithig]-d2thisp2dth2_mg[ielow][itlow]);
            hige2 = d2thisp2dth2_mg[iehig][itlow] + dt*(d2thisp2dth2_mg[iehig][ithig]-d2thisp2dth2_mg[iehig][itlow]);
            d2thisp2dth2 = lowe2 + de*(hige2-lowe2);
        }

        // Compute derivative. Careful, it's tricky too: We need to account for the dependence
        // of p0,p1,p2 on theta, but also of the dependence of R itself on theta!
        // -----------------------------------------------------------------------------------
        double Rtop2 = pow(R,thisp2);
        double Rtop21= Rtop2/R;
        double dfluxdth   = flux0 * (1./thisp0*dthisp0dth 
                                     -thisp2*Rtop21*dRdTh 
                                     -pow(R,thisp2)*log(R)*dthisp2dth);
        if (mode==3) {
            if (dfluxdth!=dfluxdth) {
                cout    << "Warning dfluxdth mg" << endl; 
                outfile << "Warning dfluxdth mg" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
                return 0.;
            }
            // return dfluxdth*FC; // account for finite size of tank
            return dfluxdth;
        } else if (mode==31) { // d^2 flux / dtheta dR
            //double d2flux_dthdR = -dfluxdth*thisp2*Rtop21 
            //                      -flux0*(
            //                              dthisp2dth*Rtop21*(1+thisp2*log(R)) 
            //                              +dRdTh*thisp2*(thisp2-1)*pow(R,thisp2-2)
            //                             );                                     
            double logR = log(R);
            double Rtop2 = pow(R,thisp2);
            double Rtop21= Rtop2/R;
            double d2flux_dthdR = flux0*(thisp2*Rtop21*dthisp0dth/thisp0 -Rtop21*dthisp2dth 
                                         -thisp2*Rtop21*(logR*dthisp2dth+(-1.+thisp2*dRdTh)/R)
                                         +thisp2*pow(Rtop2,2.)/R*(logR*dthisp2dth+thisp2*dRdTh/R));
            if (d2flux_dthdR!=d2flux_dthdR) {
                cout    << "Warning d2fluxdthdR mg" << endl; 
                outfile << "Warning d2fluxdthdR mg" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
                return 0.;
            }
            // Add correction factor for flux
            // ------------------------------
            // double dfluxdth   = flux0 * (1./thisp0*dthisp0dth 
            //                             -thisp2*Rtop21*dRdTh 
            //                             -pow(R,thisp2)*log(R)*dthisp2dth);
            // d2flux_dthdR = d2flux_dthdR*FC + dfluxdth * FluxCorr(R,2,1);
            return d2flux_dthdR;
        } else if (mode==32) { // d^2 flux / dtheta2
            // double p2dr = thisp2*dRdTh/R;
            // double exp  = exp(-pow(R,thisp2));
            double Rtop2 = pow(R,thisp2);
            double logR   = log(R);
            //double factor = -thisp2*pow(dRdTh/R,2.) + thisp2*d2RdTh2/R + 2*dRdTh*dthisp2dth/R+logR*d2thisp2dth2;
            //double d2flux_dth2  = flux0*(-2./thisp0*Rtop2*dthisp0dth*(p2dr+logR*dthisp2dth) -
            //                      Rtop2*pow(p2dr+logR*dthisp2dth,2.) + 
            //                      pow(R,2.*thisp2)*pow(p2dr+logR*dthisp2dth,2.) +
            //                      1./thisp0*d2thisp0dth2-flux0*Rtop2*factor);
            double d2flux_dth2 = flux0/thisp0*( - 2.*Rtop21*dthisp0dth*(logR*R*dthisp2dth+thisp2*dRdTh)
                                                - thisp0*Rtop21/R*pow(logR*R*dthisp2dth+thisp2*dRdTh,2.)
                                                + thisp0*pow(Rtop21,2.)*pow(logR*R*dthisp2dth+thisp2*dRdTh,2.)
                                                + d2thisp0dth2-thisp0*Rtop21/R*(-thisp2*pow(dRdTh,2.)
                                                + logR*R*R*d2thisp2dth2+R*(2.*dthisp2dth*dRdTh+thisp2*d2RdTh2)));
            if (d2flux_dth2 != d2flux_dth2) {
                cout << "Warning d2fluxdth2 mg" << endl;
                return 0.;
            }
            // return d2flux_dth2*FC; // account for size of tank
            return d2flux_dth2;
        } else {
            return 0.; 
        }
    } 
    return 0.; // If all else fails    
}

// Function parametrizing muon content in proton showers
// -----------------------------------------------------
double MFromP (double energy, double theta, double R, int mode, double dRdTh=0) {

    // Protect against out of range values
    // -----------------------------------
    if (R<Rmin) R = Rmin; 
    if (energy<0.1 || energy>10.) return 0.;
    if (theta<0. || theta>thetamax) return 0.;

    // Convert energy into the function we use in the interpolation
    // ------------------------------------------------------------
    double xe = 0.5+20.*(log(energy)-log_01)/logdif; // energy is in PeV
    double xe2 = xe*xe;

    // Interpolate the two parameters given the wanted theta
    // -----------------------------------------------------
    double thisp0, thisp2;

    int ielow = (int)(energy/0.1)-1;
    int iehig = ielow + 1;
    double de = (energy-ielow*0.1-0.1)/0.1;
    int itlow = (int)(theta/thetamax*99.);
    int ithig = itlow + 1;
    double dt = (theta - itlow*(thetamax/99.))/(thetamax/99.);
    // Handle boundaries first
    if (iehig==100) {
        if (ithig==100) { // Upper edge of grid
            thisp0 = thisp0_mp[99][99];
            thisp2 = thisp2_mp[99][99];
        } else { // Move on energy edge interpolating in theta
            thisp0 = thisp0_mp[99][itlow] + dt*(thisp0_mp[99][ithig]-thisp0_mp[99][itlow]);
            thisp2 = thisp2_mp[99][itlow] + dt*(thisp2_mp[99][ithig]-thisp2_mp[99][itlow]);
        }
    } else if (ithig==100) { // Theta edge, interpolate in energy
        thisp0 = thisp0_mp[ielow][99] + de*(thisp0_mp[iehig][99]-thisp0_mp[ielow][99]);
        thisp2 = thisp2_mp[ielow][99] + de*(thisp2_mp[iehig][99]-thisp2_mp[ielow][99]);
    } else {
        // Do 2D interpolation
        double lowe0 = thisp0_mp[ielow][itlow] + dt*(thisp0_mp[ielow][ithig]-thisp0_mp[ielow][itlow]);
        double hige0 = thisp0_mp[iehig][itlow] + dt*(thisp0_mp[iehig][ithig]-thisp0_mp[iehig][itlow]);
        thisp0       = lowe0 + de*(hige0-lowe0);
        double lowe2 = thisp2_mp[ielow][itlow] + dt*(thisp2_mp[ielow][ithig]-thisp2_mp[ielow][itlow]);
        double hige2 = thisp2_mp[iehig][itlow] + dt*(thisp2_mp[iehig][ithig]-thisp2_mp[iehig][itlow]);
        thisp2       = lowe2 + de*(hige2-lowe2);
    }

    double flux0 = TankArea*0.02*thisp0*exp(-1.*pow(R,thisp2)); // Function for muons. Note factor of 50 (different binning of original histos)
    double flux  = flux0 + fluxB_mu;
    // Compute correction factor accounting for physical size of tank
    // ------------------------------------------------------------------
    // double FC = FluxCorr(R,3,0);

    if (mode==0) { // Return function value
        // flux0 = flux0*FC; // Correct for tank size
        if (flux0>largenumber) return largenumber;
        if (flux0<epsilon2) return epsilon2;
        if (flux0!=flux0) {
            cout    << "Warning flux0 mp; E,T,R = " << energy << " " << theta << " " << R << endl;
            outfile << "Warning flux0 mp; E,T,R = " << energy << " " << theta << " " << R << endl;
            warnings2++;
            return 0.; // Protect against nans
        }
        return flux0;
    } else if (mode==1) { // Return derivative with respect to R
        double dfluxdR = -flux0 * thisp2*pow(R,thisp2-1.);  
        if (dfluxdR!=dfluxdR) {
            cout    << "Warning dfluxdR mp" << endl; 
            outfile << "Warning dfluxdR mp" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.;
        }
        // Add derivative of correction factor
        // -----------------------------------
        // dfluxdR = dfluxdR * FC + flux0*FluxCorr(R,3,1);
        return dfluxdR; 
    } else if (mode==2) { // Return derivative with respect to energy
        // Note, this is tricky: we are deriving h = A*exp(f(g(e))) over de,
        // with h = flux = TankArea*0.02*thisp0*exp(-1.*pow(R,thisp2))
        // and with A = 0.02*TankArea*thisp0, 
        //          f = -pow(x), g = x^p2
        // The dp0/de term is easy (flux/p0 dp0/de), but the other 
        // is like dh(f(g(e)))/de = h'(f(g))f'(g)g'(e),
        // For that one we get A*exp()*f(g(e))*f'(g(e))*g'(e)
        // and      A*exp = flux.
        // That is A*exp()*R^p2*logR*dp2de = flux * R^p2 logR dp2de

        // Interpolate dthisp0de, dthisp2de - Handle boundaries first
        double dthisp0de, dthisp2de;
        if (iehig==100) {
            if (ithig==100) { // Upper edge of grid
                dthisp0de = dthisp0de_mp[99][99];
                dthisp2de = dthisp2de_mp[99][99];
            } else { // Move on energy edge interpolating in theta
                dthisp0de = dthisp0de_mp[99][itlow] + dt*(dthisp0de_mp[99][ithig]-dthisp0de_mp[99][itlow]);
                dthisp2de = dthisp2de_mp[99][itlow] + dt*(dthisp2de_mp[99][ithig]-dthisp2de_mp[99][itlow]);
            }
        } else if (ithig==100) { // Theta edge, interpolate in energy
            dthisp0de = dthisp0de_mp[ielow][99] + de*(dthisp0de_mp[iehig][99]-dthisp0de_mp[ielow][99]);
            dthisp2de = dthisp2de_mp[ielow][99] + de*(dthisp2de_mp[iehig][99]-dthisp2de_mp[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0de_mp[ielow][itlow] + dt*(dthisp0de_mp[ielow][ithig]-dthisp0de_mp[ielow][itlow]);
            double hige0 = dthisp0de_mp[iehig][itlow] + dt*(dthisp0de_mp[iehig][ithig]-dthisp0de_mp[iehig][itlow]);
            dthisp0de    = lowe0 + de*(hige0-lowe0);
            double lowe2 = dthisp2de_mp[ielow][itlow] + dt*(dthisp2de_mp[ielow][ithig]-dthisp2de_mp[ielow][itlow]);
            double hige2 = dthisp2de_mp[iehig][itlow] + dt*(dthisp2de_mp[iehig][ithig]-dthisp2de_mp[iehig][itlow]);
            dthisp2de    = lowe2 + de*(hige2-lowe2);
        }

        double dfluxde = flux0*(1./thisp0*dthisp0de -pow(R,thisp2)*log(R)*dthisp2de);
        if (dfluxde!=dfluxde) {
            cout    << "Warning dfluxde mp" << endl; 
            outfile << "Warning dfluxde mp" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.;
        }
        // return dfluxde*FC; // Account for tank size
        return dfluxde;
    } else if (mode==3) { // Return derivative with respect to theta
        // Interpolate dthisp0dth, dthisp2dth - Handle boundaries first
        double dthisp0dth, dthisp2dth;
        if (iehig==100) {
            if (ithig==100) { // Upper edge of grid
                dthisp0dth = dthisp0dth_mp[99][99];
                dthisp2dth = dthisp2dth_mp[99][99];
            } else { // Move on energy edge interpolating in theta
                dthisp0dth = dthisp0dth_mp[99][itlow] + dt*(dthisp0dth_mp[99][ithig]-dthisp0dth_mp[99][itlow]);
                dthisp2dth = dthisp2dth_mp[99][itlow] + dt*(dthisp2dth_mp[99][ithig]-dthisp2dth_mp[99][itlow]);
            }
        } else if (ithig==100) { // Theta edge, interpolate in energy
            dthisp0dth = dthisp0dth_mp[ielow][99] + de*(dthisp0dth_mp[iehig][99]-dthisp0dth_mp[ielow][99]);
            dthisp2dth = dthisp2dth_mp[ielow][99] + de*(dthisp2dth_mp[iehig][99]-dthisp2dth_mp[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0dth_mp[ielow][itlow] + dt*(dthisp0dth_mp[ielow][ithig]-dthisp0dth_mp[ielow][itlow]);
            double hige0 = dthisp0dth_mp[iehig][itlow] + dt*(dthisp0dth_mp[iehig][ithig]-dthisp0dth_mp[iehig][itlow]);
            dthisp0dth   = lowe0 + de*(hige0-lowe0);
            double lowe2 = dthisp2dth_mp[ielow][itlow] + dt*(dthisp2dth_mp[ielow][ithig]-dthisp2dth_mp[ielow][itlow]);
            double hige2 = dthisp2dth_mp[iehig][itlow] + dt*(dthisp2dth_mp[iehig][ithig]-dthisp2dth_mp[iehig][itlow]);
            dthisp2dth   = lowe2 + de*(hige2-lowe2);
        }
        // Compute derivative. Careful, it's tricky too: We need to account for the dependence
        // of p0,p1,p2 on theta, but also of the dependence of R itself on theta!
        // -----------------------------------------------------------------------------------
        double dfluxdth   = flux0*(1./thisp0*dthisp0dth 
                                   -thisp2*pow(R,thisp2-1)*dRdTh 
                                   -pow(R,thisp2)*log(R)*dthisp2dth);
        if (dfluxdth!=dfluxdth) {
            cout    << "Warning dfluxdth mp" << endl; 
            outfile << "Warning dfluxdth mp" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.;
        }
        // return dfluxdth*FC; // Account for tank size
        return dfluxdth;
    } else {    
        return 0.;
    }
}

// Function parametrizing ele+gamma content in gamma showers
// ---------------------------------------------------------
double EFromG (double energy, double theta, double R, int mode, double dRdTh=0, double d2RdTh2=0) {

    // Protect against out of range values
    // -----------------------------------
    if (R<Rmin) R = Rmin; 
    if (energy<0.1 || energy>10.) return 0.;
    if (theta<0. || theta>thetamax) return 0.;

    // Convert energy into the function we use in the interpolation
    // ------------------------------------------------------------
    double xe  = 0.5+20.*(log(energy)-log_01)/logdif; // energy is in PeV
    double xe2 = xe*xe;

    // Interpolate the two parameters given the wanted theta
    // -----------------------------------------------------
    double thisp0, thisp1, thisp2;

    int ielow = (int)(energy/0.1)-1;
    int iehig = ielow + 1;
    double de = (energy-ielow*0.1-0.1)/0.1;
    int itlow = (int)(theta/thetamax*99.);
    int ithig = itlow + 1;
    double dt = (theta - itlow*(thetamax/99.))/(thetamax/99.);
    // Handle boundaries first
    if (iehig==100) {
        if (ithig==100) { // Upper edge of grid
            thisp0 = thisp0_eg[99][99];
            thisp1 = thisp1_eg[99][99];
            thisp2 = thisp2_eg[99][99];
        } else { // Move on energy edge interpolating in theta
            thisp0 = thisp0_eg[99][itlow] + dt*(thisp0_eg[99][ithig]-thisp0_eg[99][itlow]);
            thisp1 = thisp1_eg[99][itlow] + dt*(thisp1_eg[99][ithig]-thisp1_eg[99][itlow]);
            thisp2 = thisp2_eg[99][itlow] + dt*(thisp2_eg[99][ithig]-thisp2_eg[99][itlow]);
        }
    } else if (ithig==100) { // Theta edge, interpolate in energy
        thisp0 = thisp0_eg[ielow][99] + de*(thisp0_eg[iehig][99]-thisp0_eg[ielow][99]);
        thisp1 = thisp1_eg[ielow][99] + de*(thisp1_eg[iehig][99]-thisp1_eg[ielow][99]);
        thisp2 = thisp2_eg[ielow][99] + de*(thisp2_eg[iehig][99]-thisp2_eg[ielow][99]);
    } else {
        // Do 2D interpolation
        double lowe0 = thisp0_eg[ielow][itlow] + dt*(thisp0_eg[ielow][ithig]-thisp0_eg[ielow][itlow]);
        double hige0 = thisp0_eg[iehig][itlow] + dt*(thisp0_eg[iehig][ithig]-thisp0_eg[iehig][itlow]);
        thisp0       = lowe0 + de*(hige0-lowe0);
        double lowe1 = thisp1_eg[ielow][itlow] + dt*(thisp1_eg[ielow][ithig]-thisp1_eg[ielow][itlow]);
        double hige1 = thisp1_eg[iehig][itlow] + dt*(thisp1_eg[iehig][ithig]-thisp1_eg[iehig][itlow]);
        thisp1       = lowe1 + de*(hige1-lowe1);
        double lowe2 = thisp2_eg[ielow][itlow] + dt*(thisp2_eg[ielow][ithig]-thisp2_eg[ielow][itlow]);
        double hige2 = thisp2_eg[iehig][itlow] + dt*(thisp2_eg[iehig][ithig]-thisp2_eg[iehig][itlow]);
        thisp2       = lowe2 + de*(hige2-lowe2);
    }

    double flux0 = TankArea*thisp0*exp(-thisp1*pow(R,thisp2)); 
    double flux  = flux0 + fluxB_e;
    // Compute correction factor accounting for physical size of tank
    // ------------------------------------------------------------------
    // double FC = FluxCorr(R,0,0);

    if (mode==0) { // Return function value
        // flux0 = flux0*FC; // Correct for tank size
        if (flux0>largenumber) return largenumber;
        if (flux0<epsilon2) return epsilon2;
        if (flux0!=flux0) {
            cout    << "Warning flux0 eg; E,T,R = " << energy << " " << theta << " " << R << endl;
            outfile << "Warning flux0 eg; E,T,R = " << energy << " " << theta << " " << R << endl;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.; // Protect against nans
        }
        return flux0;
    } else if (mode==1) { // Return derivative with respect to R
        double dfluxdR = -flux0 * thisp1*thisp2*pow(R,thisp2-1);  
        if (dfluxdR!=dfluxdR) {
            cout    << "Warning dfluxdR eg" << endl; 
            outfile << "Warning dfluxdR eg" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.;
        }
        // Add derivative of correction factor
        // -----------------------------------
        // dfluxdR = dfluxdR * FC + flux0*FluxCorr(R,0,1);
        return dfluxdR; 
    } else if (mode==12) { // Return second derivative wrt R
        double d2fluxdR2 = flux0 * (pow(thisp1*thisp2,2.)*pow(R,2.*thisp2-2.)-thisp1*thisp2*(thisp2-1.)*pow(R,thisp2-2));
        if (d2fluxdR2!=d2fluxdR2) {
            cout    << "Warning d2fluxdR2 eg" << endl; 
            outfile << "Warning d2fluxdR2 eg" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.;
        }
        // Add derivative of correction factor
        // -----------------------------------
        // double dfluxdR = -flux0 * thisp1*thisp2*pow(R,thisp2-1);  
        // d2fluxdR2 = d2fluxdR2 * FC + 2.*dfluxdR*FluxCorr(R,0,1) + flux0*FluxCorr(R,0,2);
        return d2fluxdR2; 
    } else if (mode==2) { // Return derivative with respect to energy
        // Note, this is tricky: we are deriving A*exp(f(g(e))) over de.
        // This is like dh(g(f(e)))/de = h'(g(f))g'(f)f'(e),
        // So we get A*exp()*f(g(e))*f'(g(e))*g'(e)
        // That is A*exp()*(-p1*R^p2)*logR*dp2de = flux * -p1*R^p2 logR dp2de
        // flux = A(e)*exp(-p1(e)*R^p2(e))
        // dflux/de = flux/A *dA/de + flux*[ -R^p2(e)*dp1/de -p1(e)R^p2(e) logR dp2de] 

        // Interpolate dthisp0de, dthisp2de - Handle boundaries first
        double dthisp0de, dthisp1de, dthisp2de;
        if (iehig==100) {
            if (ithig==100) { // Upper edge of grid
                dthisp0de = dthisp0de_eg[99][99];
                dthisp1de = dthisp1de_eg[99][99];
                dthisp2de = dthisp2de_eg[99][99];
            } else { // Move on energy edge interpolating in theta
                dthisp0de = dthisp0de_eg[99][itlow] + dt*(dthisp0de_eg[99][ithig]-dthisp0de_eg[99][itlow]);
                dthisp1de = dthisp1de_eg[99][itlow] + dt*(dthisp1de_eg[99][ithig]-dthisp1de_eg[99][itlow]);
                dthisp2de = dthisp2de_eg[99][itlow] + dt*(dthisp2de_eg[99][ithig]-dthisp2de_eg[99][itlow]);
            }
        } else if (ithig==100) { // Theta edge, interpolate in energy
            dthisp0de = dthisp0de_eg[ielow][99] + de*(dthisp0de_eg[iehig][99]-dthisp0de_eg[ielow][99]);
            dthisp1de = dthisp1de_eg[ielow][99] + de*(dthisp1de_eg[iehig][99]-dthisp1de_eg[ielow][99]);
            dthisp2de = dthisp2de_eg[ielow][99] + de*(dthisp2de_eg[iehig][99]-dthisp2de_eg[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0de_eg[ielow][itlow] + dt*(dthisp0de_eg[ielow][ithig]-dthisp0de_eg[ielow][itlow]);
            double hige0 = dthisp0de_eg[iehig][itlow] + dt*(dthisp0de_eg[iehig][ithig]-dthisp0de_eg[iehig][itlow]);
            dthisp0de    = lowe0 + de*(hige0-lowe0);
            double lowe1 = dthisp1de_eg[ielow][itlow] + dt*(dthisp1de_eg[ielow][ithig]-dthisp1de_eg[ielow][itlow]);
            double hige1 = dthisp1de_eg[iehig][itlow] + dt*(dthisp1de_eg[iehig][ithig]-dthisp1de_eg[iehig][itlow]);
            dthisp1de    = lowe1 + de*(hige1-lowe1);
            double lowe2 = dthisp2de_eg[ielow][itlow] + dt*(dthisp2de_eg[ielow][ithig]-dthisp2de_eg[ielow][itlow]);
            double hige2 = dthisp2de_eg[iehig][itlow] + dt*(dthisp2de_eg[iehig][ithig]-dthisp2de_eg[iehig][itlow]);
            dthisp2de    = lowe2 + de*(hige2-lowe2);
        }

        double dfluxde = flux0*(1./thisp0*dthisp0de -pow(R,thisp2)*dthisp1de -thisp1*pow(R,thisp2)*log(R)*dthisp2de);
        if (dfluxde!=dfluxde) {
            cout    << "Warning dfluxde eg" << endl; 
            outfile << "Warning dfluxde eg" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.;
        }
        // return dfluxde*FC; // Consider flux correction
        return dfluxde; 
    } else if (mode==22 || mode==23 || mode==24 || mode==25) { // Return other derivatives wrt energy and distance
        // 2 = dflux/de
        // 22= d2flux/de2
        // 23= d2flux/dedr
        // 24= d3flux/de2dr
        // 25= d3flux/de3
        // Interpolate dthisp0de, dthisp2de - Handle boundaries first
        double dthisp0de, dthisp1de, dthisp2de;
        double d2thisp0de2, d2thisp1de2, d2thisp2de2;
        double d3thisp0de3, d3thisp1de3, d3thisp2de3;
        if (iehig==100) {
            if (ithig==100) { // Upper edge of grid
                dthisp0de = dthisp0de_eg[99][99];
                dthisp1de = dthisp1de_eg[99][99];
                dthisp2de = dthisp2de_eg[99][99];
                d2thisp0de2 = d2thisp0de2_eg[99][99];
                d2thisp1de2 = d2thisp1de2_eg[99][99];
                d2thisp2de2 = d2thisp2de2_eg[99][99];
                d3thisp0de3 = d3thisp0de3_eg[99][99];
                d3thisp1de3 = d3thisp1de3_eg[99][99];
                d3thisp2de3 = d3thisp2de3_eg[99][99];
            } else { // Move on energy edge interpolating in theta
                dthisp0de = dthisp0de_eg[99][itlow] + dt*(dthisp0de_eg[99][ithig]-dthisp0de_eg[99][itlow]);
                dthisp1de = dthisp1de_eg[99][itlow] + dt*(dthisp1de_eg[99][ithig]-dthisp1de_eg[99][itlow]);
                dthisp2de = dthisp2de_eg[99][itlow] + dt*(dthisp2de_eg[99][ithig]-dthisp2de_eg[99][itlow]);
                d2thisp0de2 = d2thisp0de2_eg[99][itlow] + dt*(d2thisp0de2_eg[99][ithig]-d2thisp0de2_eg[99][itlow]);
                d2thisp1de2 = d2thisp1de2_eg[99][itlow] + dt*(d2thisp1de2_eg[99][ithig]-d2thisp1de2_eg[99][itlow]);
                d2thisp2de2 = d2thisp2de2_eg[99][itlow] + dt*(d2thisp2de2_eg[99][ithig]-d2thisp2de2_eg[99][itlow]);
                d3thisp0de3 = d3thisp0de3_eg[99][itlow] + dt*(d3thisp0de3_eg[99][ithig]-d3thisp0de3_eg[99][itlow]);
                d3thisp1de3 = d3thisp1de3_eg[99][itlow] + dt*(d3thisp1de3_eg[99][ithig]-d3thisp1de3_eg[99][itlow]);
                d3thisp2de3 = d3thisp2de3_eg[99][itlow] + dt*(d3thisp2de3_eg[99][ithig]-d3thisp2de3_eg[99][itlow]);
            }
        } else if (ithig==100) { // Theta edge, interpolate in energy
            dthisp0de = dthisp0de_eg[ielow][99] + de*(dthisp0de_eg[iehig][99]-dthisp0de_eg[ielow][99]);
            dthisp1de = dthisp1de_eg[ielow][99] + de*(dthisp1de_eg[iehig][99]-dthisp1de_eg[ielow][99]);
            dthisp2de = dthisp2de_eg[ielow][99] + de*(dthisp2de_eg[iehig][99]-dthisp2de_eg[ielow][99]);
            d2thisp0de2 = d2thisp0de2_eg[ielow][99] + de*(d2thisp0de2_eg[iehig][99]-d2thisp0de2_eg[ielow][99]);
            d2thisp1de2 = d2thisp1de2_eg[ielow][99] + de*(d2thisp1de2_eg[iehig][99]-d2thisp1de2_eg[ielow][99]);
            d2thisp2de2 = d2thisp2de2_eg[ielow][99] + de*(d2thisp2de2_eg[iehig][99]-d2thisp2de2_eg[ielow][99]);
            d3thisp0de3 = d3thisp0de3_eg[ielow][99] + de*(d3thisp0de3_eg[iehig][99]-d3thisp0de3_eg[ielow][99]);
            d3thisp1de3 = d3thisp1de3_eg[ielow][99] + de*(d3thisp1de3_eg[iehig][99]-d3thisp1de3_eg[ielow][99]);
            d3thisp2de3 = d3thisp2de3_eg[ielow][99] + de*(d3thisp2de3_eg[iehig][99]-d3thisp2de3_eg[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0de_eg[ielow][itlow] + dt*(dthisp0de_eg[ielow][ithig]-dthisp0de_eg[ielow][itlow]);
            double hige0 = dthisp0de_eg[iehig][itlow] + dt*(dthisp0de_eg[iehig][ithig]-dthisp0de_eg[iehig][itlow]);
            dthisp0de    = lowe0 + de*(hige0-lowe0);
            double lowe1 = dthisp1de_eg[ielow][itlow] + dt*(dthisp1de_eg[ielow][ithig]-dthisp1de_eg[ielow][itlow]);
            double hige1 = dthisp1de_eg[iehig][itlow] + dt*(dthisp1de_eg[iehig][ithig]-dthisp1de_eg[iehig][itlow]);
            dthisp1de    = lowe1 + de*(hige1-lowe1);
            double lowe2 = dthisp2de_eg[ielow][itlow] + dt*(dthisp2de_eg[ielow][ithig]-dthisp2de_eg[ielow][itlow]);
            double hige2 = dthisp2de_eg[iehig][itlow] + dt*(dthisp2de_eg[iehig][ithig]-dthisp2de_eg[iehig][itlow]);
            dthisp2de    = lowe2 + de*(hige2-lowe2);
            lowe0 = d2thisp0de2_eg[ielow][itlow] + dt*(d2thisp0de2_eg[ielow][ithig]-d2thisp0de2_eg[ielow][itlow]);
            hige0 = d2thisp0de2_eg[iehig][itlow] + dt*(d2thisp0de2_eg[iehig][ithig]-d2thisp0de2_eg[iehig][itlow]);
            d2thisp0de2    = lowe0 + de*(hige0-lowe0);
            lowe1 = d2thisp1de2_eg[ielow][itlow] + dt*(d2thisp1de2_eg[ielow][ithig]-d2thisp1de2_eg[ielow][itlow]);
            hige1 = d2thisp1de2_eg[iehig][itlow] + dt*(d2thisp1de2_eg[iehig][ithig]-d2thisp1de2_eg[iehig][itlow]);
            d2thisp1de2    = lowe1 + de*(hige1-lowe1);
            lowe2 = d2thisp2de2_eg[ielow][itlow] + dt*(d2thisp2de2_eg[ielow][ithig]-d2thisp2de2_eg[ielow][itlow]);
            hige2 = d2thisp2de2_eg[iehig][itlow] + dt*(d2thisp2de2_eg[iehig][ithig]-d2thisp2de2_eg[iehig][itlow]);
            d2thisp2de2    = lowe2 + de*(hige2-lowe2);
            lowe0 = d3thisp0de3_eg[ielow][itlow] + dt*(d3thisp0de3_eg[ielow][ithig]-d3thisp0de3_eg[ielow][itlow]);
            hige0 = d3thisp0de3_eg[iehig][itlow] + dt*(d3thisp0de3_eg[iehig][ithig]-d3thisp0de3_eg[iehig][itlow]);
            d3thisp0de3    = lowe0 + de*(hige0-lowe0);
            lowe1 = d3thisp1de3_eg[ielow][itlow] + dt*(d3thisp1de3_eg[ielow][ithig]-d3thisp1de3_eg[ielow][itlow]);
            hige1 = d3thisp1de3_eg[iehig][itlow] + dt*(d3thisp1de3_eg[iehig][ithig]-d3thisp1de3_eg[iehig][itlow]);
            d3thisp1de3    = lowe1 + de*(hige1-lowe1);
            lowe2 = d3thisp2de3_eg[ielow][itlow] + dt*(d3thisp2de3_eg[ielow][ithig]-d3thisp2de3_eg[ielow][itlow]);
            hige2 = d3thisp2de3_eg[iehig][itlow] + dt*(d3thisp2de3_eg[iehig][ithig]-d3thisp2de3_eg[iehig][itlow]);
            d3thisp2de3    = lowe2 + de*(hige2-lowe2);
        }
        if (mode==22) { // d2flux/de2 term
            double Rtop2m1 = pow(R,thisp2-1.);
            double logR = log(R);
            double Rtop2 = pow(R,thisp2);
            // double Cfactor = dthisp0de/thisp0 -Rtop2*dthisp1de -thisp1*Rtop2*logR*dthisp2de;
            // double dCdefactor = -pow(dthisp0de/thisp0,2.) + d2thisp0de2/thisp0 
            //                   -Rtop2*logR*dthisp2de*dthisp1de - Rtop2*d2thisp1de2
            //                   -logR*(dthisp1de*Rtop2*dthisp2de+thisp1*Rtop2*logR*pow(dthisp2de,2.)+thisp1*Rtop2*d2thisp2de2);
            //double dfluxde = flux0 * Cfactor;
            //double d2fluxde2 = dfluxde*Cfactor + flux0 * dCdefactor;
            //cout << Cfactor << " " << dCdefactor << " " << flux << " " << d2fluxde2 << endl;
            //double d2fluxde2 = flux0/thisp0*(-2.*Rtop2*dthisp0de*(dthisp1de+logR*thisp1*dthisp2de)
            //                                 +d2thisp0de2+Rtop2*thisp0*(Rtop2*pow(dthisp1de,2.)+2.*logR*(-1.+Rtop2*thisp1)*dthisp1de*dthisp2de
            //                                 +Rtop2*pow(logR*thisp1*dthisp2de,2.)-d2thisp1de2-logR*thisp1*(logR*pow(dthisp2de,2.)+d2thisp2de2)));
            // Mathematica sol, see d3f0mde3.nb eq. 7
            // ------------------------------------------------------------------------------------------------------------------------------------
            double d2fluxde2 = flux0 * (2./thisp0*dthisp0de*(-Rtop2*dthisp1de-Rtop2*logR*thisp1*dthisp2de) 
                               + pow(-Rtop2*dthisp1de-Rtop2*logR*thisp1*dthisp2de,2.)
                               + 1./thisp0*d2thisp0de2 
                               + (-2.*Rtop2*logR*dthisp1de*dthisp2de
                                  -Rtop2*thisp1*pow(logR*dthisp2de,2.)
                                  -Rtop2*d2thisp1de2-Rtop2*logR*thisp1*d2thisp2de2));
            if (d2fluxde2!=d2fluxde2) {
                cout    << "Warning d2fluxde2 eg" << endl; 
                outfile << "Warning d2fluxde2 eg" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
                return 0.;
            }
            // return d2fluxde2*FC; // Include flux correction for tank size
            return d2fluxde2;
        } else if (mode==23) { // d2flux/drde term
            double logR = log(R);
            double Rtop2 = pow(R,thisp2);
            double Rtop2m1 = pow(R,thisp2-1);
            //double dfluxdr = -flux0 * thisp1*thisp2*Rtop2m1;  
            //double Dfactor = -thisp2*Rtop2m1*dthisp1de-thisp1*dthisp2de*(thisp2*Rtop2m1*logR+Rtop2m1);
            //double Cfactor = dthisp0de/thisp0 -Rtop2*dthisp1de -thisp1*Rtop2*logR*dthisp2de;
            //double d2fluxdedr = dfluxdr*Cfactor + flux0*Dfactor;
            double d2fluxdedr = flux0/thisp0*Rtop2m1*(-thisp0*thisp2*dthisp1de+Rtop2*logR*thisp0*pow(thisp1,2.)*thisp2*dthisp2de
                                                      -thisp1*(thisp0*dthisp2de+thisp2*(dthisp0de-Rtop2*thisp0*dthisp1de+logR*thisp0*dthisp2de)));
            
            if (d2fluxdedr!=d2fluxdedr) {
                cout    << "Warning d2fluxde2 eg" << endl; 
                outfile << "Warning d2fluxde2 eg" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
                return 0.;
            }
            // For the corrected flux we need to take the df/de and consider also the derivative of fc over r
            // ----------------------------------------------------------------------------------------------
            // double dfluxde = flux0*(1./thisp0*dthisp0de -pow(R,thisp2)*dthisp1de -thisp1*pow(R,thisp2)*log(R)*dthisp2de);
            // d2fluxdedr = d2fluxdedr*FC + dfluxde*FluxCorr(R,0,1);
            return d2fluxdedr;
        } else if (mode==24) {
            double logR = log(R);
            double Rtop2 = pow(R,thisp2);
            double Rtop2m1 = pow(R,thisp2-1);
            //double Dfactor = -thisp2*Rtop2m1*dthisp1de-thisp1*dthisp2de*(thisp2*Rtop2m1*logR+Rtop2m1);
            //double dDfactorde = -Rtop2m1*(dthisp1de*dthisp2de+thisp2*logR*dthisp1de+thisp2*d2thisp1de2)
            //                    -Rtop2m1*(dthisp1de*dthisp2de+thisp1*d2thisp2de2)*(thisp2*logR+1.)
            //                    -Rtop2m1*logR*thisp1*pow(dthisp2de,2.)*(thisp2*logR+2.);    
            //double Cfactor = dthisp0de/thisp0 -Rtop2*dthisp1de -thisp1*Rtop2*logR*dthisp2de;
            //double dCdefactor = -pow(dthisp0de/thisp0,2.) + d2thisp0de2/thisp0 
            //                    -Rtop2*logR*dthisp2de*dthisp1de - Rtop2*d2thisp1de2
            //                    -logR*(dthisp1de*Rtop2*dthisp2de+thisp2*Rtop2*logR*d2thisp2de2+thisp1*Rtop2*d2thisp2de2);
            //double dfluxdr = -flux0 * thisp1*thisp2*Rtop2m1;
            //double dfluxde = flux0 * Cfactor;
            //double d3fluxdrde2 = (dfluxdr*Cfactor + flux0*Dfactor) * Cfactor
            //                     + dfluxdr*dCdefactor + dfluxde*Dfactor + flux0*dDfactorde;
            double d3fluxdrde2 = -flux0/thisp0*Rtop2m1*(2.*thisp0*dthisp1de*dthisp2de-2.*Rtop2*logR*thisp0*pow(thisp1*dthisp2de,2.)
                                                        + thisp1*(2.*dthisp0de*dthisp2de+thisp0*(-2.*Rtop2*dthisp1de*dthisp2de+2.*logR*pow(dthisp2de,2.)+d2thisp2de2))
                                                        + thisp2*(-2.*(-1.+Rtop2*thisp1)*dthisp0de*(dthisp1de+logR*thisp1*dthisp2de)+thisp1*d2thisp0de2
                                                        + thisp0*(Rtop2*(-2.+Rtop2*thisp1)*pow(dthisp1de,2.)+2.*logR*(1.-3.*Rtop2*thisp1+pow(Rtop2*thisp1,2.))
                                                        *dthisp1de*dthisp2de + pow(Rtop2*logR*dthisp2de,2.)*pow(thisp1,3.)+d2thisp1de2-Rtop2*logR*pow(thisp1,2.)
                                                        *(3.*logR*pow(dthisp2de,2.)+d2thisp2de2)+thisp1*(pow(logR*dthisp2de,2.)-Rtop2*d2thisp1de2+logR*d2thisp2de2))));
            if (d3fluxdrde2!=d3fluxdrde2) {
                cout    << "Warning d3fluxdrde2 eg" << endl; 
                outfile << "Warning d3fluxdrde2 eg" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
                return 0.;
            }
            // To compute the corrected flux we need to call in the derivative of the uncorrected one:
            // d^3f'/de^2dr = d^3f/de^2dr * fc + d^2f/de^2 * dfc/dr
            // --------------------------------------------------------------------------------------- 
            // double d2fluxde2 = flux0 * (2./thisp0*dthisp0de*(-Rtop2*dthisp1de-Rtop2*logR*thisp1*dthisp2de) 
            //                    + pow(-Rtop2*dthisp1de-Rtop2*logR*thisp1*dthisp2de,2.)
            //                    + 1./thisp0*d2thisp0de2 
            //                    + (-2.*Rtop2*logR*dthisp1de*dthisp2de
            //                       -Rtop2*thisp1*pow(logR*dthisp2de,2.)
            //                       -Rtop2*d2thisp1de2-Rtop2*logR*thisp1*d2thisp2de2));
            // d3fluxdrde2 = d3fluxdrde2*FC + d2fluxde2*FluxCorr(R,0,1);
            // ----------------------------------------------------------------------------------------------
            return d3fluxdrde2;
        } else if (mode==25) { // d3flux/de3 term
            double L = log(R);
            double L2 = L*L;
            double L3 = L2*L;
            double Rtop2 = pow(R,thisp2);
            // See mathematica d3f0mde3.nb, eq. 9
            // ----------------------------------
            double d3fluxde3 = exp(-Rtop2*thisp1)*(-3.*Rtop2*dthisp1de*d2thisp0de2 - 3.*L*Rtop2*thisp1*dthisp2de*d2thisp0de2
                                                   +3.*Rtop2*dthisp0de*(Rtop2*pow(dthisp1de,2.)+2.*L*(-1.+Rtop2*thisp1)*dthisp1de*dthisp2de 
                                                   +L2*Rtop2*pow(thisp1*dthisp2de,2.)-d2thisp1de2-L*thisp1*(L*pow(dthisp2de,2.)+d2thisp2de2))
                                                   +d3thisp0de3-Rtop2*thisp0*(pow(Rtop2,2.)*pow(dthisp1de,3.)+3.*L*Rtop2*(-2.+Rtop2*thisp1)*pow(dthisp1de,2.)*dthisp2de
                                                   +L3*pow(Rtop2,2.)*pow(thisp1*dthisp2de,3.)+3.*L*dthisp2de*d2thisp1de2
                                                   -3.*L2*Rtop2*pow(thisp1,2.)*dthisp2de*(L*pow(dthisp2de,2.)+d2thisp2de2) + 3.*dthisp1de*(L2*(1.-3.*Rtop2*thisp1+pow(Rtop2*thisp1,2.))*pow(dthisp2de,2)-Rtop2*d2thisp1de2
                                                   -L*(-1.+Rtop2*thisp1)*d2thisp2de2) + d3thisp1de3+L*thisp1*(L2*pow(dthisp2de,3.)+dthisp2de*(-3.*Rtop2*d2thisp1de2+3.*L*d2thisp2de2)+d3thisp2de3)));  
            if (d3fluxde3!=d3fluxde3) {
                cout    << "Warning d3fluxde3 eg" << endl; 
                outfile << "Warning d3fluxde3 eg" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
                return 0.;
            }
            // return d3fluxde3*FC; // Account for finite size of tank
            return d3fluxde3;
        }
    } else if (mode==3 || mode==31 || mode==32) { // Return derivative with respect to theta
        // mode = 3 dflux/dtheta; mode=31 d^2flux/dthetadR

        // Interpolate dthisp0dth, dthisp2dth - Handle boundaries first
        double dthisp0dth, dthisp1dth, dthisp2dth;
        double d2thisp0dth2, d2thisp1dth2, d2thisp2dth2;
        if (iehig==100) {
            if (ithig==100) { // Upper edge of grid
                dthisp0dth = dthisp0dth_eg[99][99];
                dthisp1dth = dthisp1dth_eg[99][99];
                dthisp2dth = dthisp2dth_eg[99][99];
                d2thisp0dth2 = d2thisp0dth2_eg[99][99];
                d2thisp1dth2 = d2thisp1dth2_eg[99][99];
                d2thisp2dth2 = d2thisp2dth2_eg[99][99];
            } else { // Move on energy edge interpolating in theta
                dthisp0dth = dthisp0dth_eg[99][itlow] + dt*(dthisp0dth_eg[99][ithig]-dthisp0dth_eg[99][itlow]);
                dthisp1dth = dthisp1dth_eg[99][itlow] + dt*(dthisp1dth_eg[99][ithig]-dthisp1dth_eg[99][itlow]);
                dthisp2dth = dthisp2dth_eg[99][itlow] + dt*(dthisp2dth_eg[99][ithig]-dthisp2dth_eg[99][itlow]);
                d2thisp0dth2 = d2thisp0dth2_eg[99][itlow] + dt*(d2thisp0dth2_eg[99][ithig]-d2thisp0dth2_eg[99][itlow]);
                d2thisp1dth2 = d2thisp1dth2_eg[99][itlow] + dt*(d2thisp1dth2_eg[99][ithig]-d2thisp1dth2_eg[99][itlow]);
                d2thisp2dth2 = d2thisp2dth2_eg[99][itlow] + dt*(d2thisp2dth2_eg[99][ithig]-d2thisp2dth2_eg[99][itlow]);
            }
        } else if (ithig==100) { // Theta edge, interpolate in energy
            dthisp0dth = dthisp0dth_eg[ielow][99] + de*(dthisp0dth_eg[iehig][99]-dthisp0dth_eg[ielow][99]);
            dthisp1dth = dthisp1dth_eg[ielow][99] + de*(dthisp1dth_eg[iehig][99]-dthisp1dth_eg[ielow][99]);
            dthisp2dth = dthisp2dth_eg[ielow][99] + de*(dthisp2dth_eg[iehig][99]-dthisp2dth_eg[ielow][99]);
            d2thisp0dth2 = d2thisp0dth2_eg[ielow][99] + de*(d2thisp0dth2_eg[iehig][99]-d2thisp0dth2_eg[ielow][99]);
            d2thisp1dth2 = d2thisp1dth2_eg[ielow][99] + de*(d2thisp1dth2_eg[iehig][99]-d2thisp1dth2_eg[ielow][99]);
            d2thisp2dth2 = d2thisp2dth2_eg[ielow][99] + de*(d2thisp2dth2_eg[iehig][99]-d2thisp2dth2_eg[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0dth_eg[ielow][itlow] + dt*(dthisp0dth_eg[ielow][ithig]-dthisp0dth_eg[ielow][itlow]);
            double hige0 = dthisp0dth_eg[iehig][itlow] + dt*(dthisp0dth_eg[iehig][ithig]-dthisp0dth_eg[iehig][itlow]);
            dthisp0dth   = lowe0 + de*(hige0-lowe0);
            double lowe1 = dthisp1dth_eg[ielow][itlow] + dt*(dthisp1dth_eg[ielow][ithig]-dthisp1dth_eg[ielow][itlow]);
            double hige1 = dthisp1dth_eg[iehig][itlow] + dt*(dthisp1dth_eg[iehig][ithig]-dthisp1dth_eg[iehig][itlow]);
            dthisp1dth   = lowe1 + de*(hige1-lowe1);
            double lowe2 = dthisp2dth_eg[ielow][itlow] + dt*(dthisp2dth_eg[ielow][ithig]-dthisp2dth_eg[ielow][itlow]);
            double hige2 = dthisp2dth_eg[iehig][itlow] + dt*(dthisp2dth_eg[iehig][ithig]-dthisp2dth_eg[iehig][itlow]);
            dthisp2dth   = lowe2 + de*(hige2-lowe2);
            lowe0 = d2thisp0dth2_eg[ielow][itlow] + dt*(d2thisp0dth2_eg[ielow][ithig]-d2thisp0dth2_eg[ielow][itlow]);
            hige0 = d2thisp0dth2_eg[iehig][itlow] + dt*(d2thisp0dth2_eg[iehig][ithig]-d2thisp0dth2_eg[iehig][itlow]);
            d2thisp0dth2   = lowe0 + de*(hige0-lowe0);
            lowe1 = d2thisp1dth2_eg[ielow][itlow] + dt*(d2thisp1dth2_eg[ielow][ithig]-d2thisp1dth2_eg[ielow][itlow]);
            hige1 = d2thisp1dth2_eg[iehig][itlow] + dt*(d2thisp1dth2_eg[iehig][ithig]-d2thisp1dth2_eg[iehig][itlow]);
            d2thisp1dth2   = lowe1 + de*(hige1-lowe1);
            lowe2 = d2thisp2dth2_eg[ielow][itlow] + dt*(d2thisp2dth2_eg[ielow][ithig]-d2thisp2dth2_eg[ielow][itlow]);
            hige2 = d2thisp2dth2_eg[iehig][itlow] + dt*(d2thisp2dth2_eg[iehig][ithig]-d2thisp2dth2_eg[iehig][itlow]);
            d2thisp2dth2   = lowe2 + de*(hige2-lowe2);
        }
        // Compute derivative. Careful, it's tricky too: We need to account for the dependence
        // of p0,p1,p2 on theta, but also of the dependence of R itself on theta!
        // -----------------------------------------------------------------------------------
        double Rtop2 = pow(R,thisp2);
        double Rtop21= pow(R,thisp2-1);
        double dfluxdth   = flux0*(1./thisp0*dthisp0dth 
                                   -thisp1*thisp2*Rtop21*dRdTh 
                                   -thisp1*Rtop2*log(R)*dthisp2dth
                                   -Rtop2*dthisp1dth);
        if (mode==3) {
            if (dfluxdth!=dfluxdth) {
                cout    << "Warning dfluxdth eg" << endl; 
                outfile << "Warning dfluxdth eg" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
                return 0.;
            }
            // return dfluxdth*FC; // Account for finite size of tank
            return dfluxdth;
        } else if (mode==31) { // d^2 flux / dtheta dR
            //double d2flux_dthdR = -dfluxdth*(thisp1*thisp2*Rtop21) 
            //                      -flux0*(dthisp1dth*thisp2*Rtop21 
            //                              +dthisp2dth*(thisp1*Rtop21+thisp1*thisp2*Rtop21*log(R)) 
            //                              +dRdTh*thisp1*thisp2*(thisp2-1)*pow(R,thisp2-2) 
            //                             );
            double logR = log(R);
            double d2flux_dthdR = flux0/thisp0*Rtop21/R*(-thisp0*thisp2*R*dthisp1dth
                                                         + thisp0*pow(thisp1,2.)*thisp2*Rtop2*(logR*R*dthisp2dth+thisp2*dRdTh)
                                                         - thisp1*(thisp0*R*dthisp2dth+thisp0*pow(thisp2,2.)*dRdTh+thisp2*(-thisp0*Rtop21
                                                         *dthisp1dth+R*(dthisp0dth+logR*thisp0*dthisp2dth)-thisp0*dRdTh)));
            //if (d2flux_dthdR_math!=0) cout << d2flux_dthdR/d2flux_dthdR_math << endl;
            if (d2flux_dthdR!=d2flux_dthdR) {
                cout    << "Warning d2fluxdthdR eg" << endl; 
                outfile << "Warning d2fluxdthdR eg" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
                return 0.;
            }
            // Add correction factor
            // ---------------------
            // double dfluxdth   = flux0*(1./thisp0*dthisp0dth 
            //                        -thisp1*thisp2*Rtop21*dRdTh 
            //                        -thisp1*Rtop2*log(R)*dthisp2dth
            //                        -Rtop2*dthisp1dth);
            // d2flux_dthdR = d2flux_dthdR*FC + dfluxdth * FluxCorr(R,0,1);
            return d2flux_dthdR;
        } else if (mode==32) { // d^2 flux / dtheta2  
            //double p2dr = thisp2*dRdTh/R;
            double Rtop2 = pow(R,thisp2);
            //double expo  = exp(-thisp1*pow(R,thisp2));
            double logR  = log(R);
            //double factor = 2.*dthisp2dth*dRdTh/R - thisp2*pow(dRdTh/R,2.) + L*d2thisp2dth2 + thisp2*d2RdTh2/R;
            double frac =  thisp2*dRdTh/R;
            //double d2flux_dth2  = 2.*expo*dthisp0dth*(-Rtop2*dthisp2dth-thisp2*Rtop2*(p2dr+L*dthisp2dth)) +
            //                      expo*thisp0*pow(-Rtop2*dthisp2dth-thisp1*Rtop2*(p2dr+L*dthisp2dth),2.) +
            //                      expo*d2thisp0dth2 + expo*thisp0*(-2.*Rtop2*dthisp1dth*(p2dr+L*dthisp2dth) - 
            //                      thisp1*Rtop2*pow(p2dr+L*dthisp2dth,2.) - Rtop2*d2thisp1dth2 -thisp1*Rtop2*factor);
            double d2flux_dth2 = flux0/thisp0* (2.*dthisp0dth*(-Rtop2*dthisp1dth-thisp1*Rtop2*(frac+logR*dthisp2dth))
                                                     + thisp0*pow(-Rtop2*dthisp1dth-thisp1*Rtop2*(frac+logR*dthisp2dth),2.)               
                                                     +d2thisp0dth2 +thisp0*(-2.*Rtop2*dthisp1dth*(frac+logR*dthisp2dth)-thisp1
                                                     *Rtop2*pow(frac+logR*dthisp2dth,2.)-Rtop2*d2thisp1dth2-thisp1*Rtop2
                                                     *(2.*dthisp2dth*dRdTh/R-thisp2*pow(dRdTh/R,2.)+logR*d2thisp2dth2+thisp2*d2RdTh2/R)));
            if (d2flux_dth2 != d2flux_dth2) {
                cout    << "Warning d2fluxdth2 eg" << endl;
                outfile << "Warning d2fluxdth2 eg" << endl;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
                return 0.;
            }
            // return d2flux_dth2*FC; // Account for size of tank
            return d2flux_dth2;
        } else {
            return 0.; 
        }
    }    
    return 0.; // If all else fails    
}

// Function parametrizing ele+gamma content in proton showers
// ----------------------------------------------------------
double EFromP (double energy, double theta, double R, int mode, double dRdTh=0) {

    // Protect against out of range values
    // -----------------------------------
    if (R<Rmin) R = Rmin; 
    if (energy<0.1 || energy>10.) return 0.;
    if (theta<0. || theta>thetamax) return 0.;

    // Convert energy into the function we use in the interpolation
    // ------------------------------------------------------------
    double xe = 0.5+20.*(log(energy)-log_01)/logdif; // energy is in PeV
    double xe2 = xe*xe;

    // Interpolate the two parameters given the wanted theta
    // -----------------------------------------------------
    double thisp0, thisp1, thisp2;

    int ielow = (int)(energy/0.1)-1;
    int iehig = ielow + 1;
    double de = (energy-ielow*0.1-0.1)/0.1;
    int itlow = (int)(theta/thetamax*99.);
    int ithig = itlow + 1;
    double dt = (theta - itlow*(thetamax/99.))/(thetamax/99.);
    // Handle boundaries first
    if (iehig==100) {
        if (ithig==100) { // Upper edge of grid
            thisp0 = thisp0_ep[99][99];
            thisp1 = thisp1_ep[99][99];
            thisp2 = thisp2_ep[99][99];
        } else { // Move on energy edge interpolating in theta
            thisp0 = thisp0_ep[99][itlow]+ dt*(thisp0_ep[99][ithig]-thisp0_ep[99][itlow]);
            thisp1 = thisp1_ep[99][itlow]+ dt*(thisp1_ep[99][ithig]-thisp1_ep[99][itlow]);
            thisp2 = thisp2_ep[99][itlow]+ dt*(thisp2_ep[99][ithig]-thisp2_ep[99][itlow]);
        }
    } else if (ithig==100) { // Theta edge, interpolate in energy
        thisp0 = thisp0_ep[ielow][99]+ de*(thisp0_ep[iehig][99]-thisp0_ep[ielow][99]);
        thisp1 = thisp1_ep[ielow][99]+ de*(thisp1_ep[iehig][99]-thisp1_ep[ielow][99]);
        thisp2 = thisp2_ep[ielow][99]+ de*(thisp2_ep[iehig][99]-thisp2_ep[ielow][99]);
    } else {
        // Do 2D interpolation
        double lowe0 = thisp0_ep[ielow][itlow]+ dt*(thisp0_ep[ielow][ithig]-thisp0_ep[ielow][itlow]);
        double hige0 = thisp0_ep[iehig][itlow]+ dt*(thisp0_ep[iehig][ithig]-thisp0_ep[iehig][itlow]);
        thisp0       = lowe0 + de*(hige0-lowe0);
        double lowe1 = thisp1_ep[ielow][itlow]+ dt*(thisp1_ep[ielow][ithig]-thisp1_ep[ielow][itlow]);
        double hige1 = thisp1_ep[iehig][itlow]+ dt*(thisp1_ep[iehig][ithig]-thisp1_ep[iehig][itlow]);
        thisp1       = lowe1 + de*(hige1-lowe1);
        double lowe2 = thisp2_ep[ielow][itlow]+ dt*(thisp2_ep[ielow][ithig]-thisp2_ep[ielow][itlow]);
        double hige2 = thisp2_ep[iehig][itlow]+ dt*(thisp2_ep[iehig][ithig]-thisp2_ep[iehig][itlow]);
        thisp2       = lowe2 + de*(hige2-lowe2);
    }

    double flux0 = TankArea*thisp0*exp(-thisp1*pow(R,thisp2)); 
    double flux  = flux0 + fluxB_e;
    // Compute correction factor accounting for physical size of tank
    // ------------------------------------------------------------------
    // double FC = FluxCorr(R,1,0);

    if (mode==0) { // Return function value
        // flux0 = flux0*FC; // Account for tank size
        if (flux0>largenumber) return largenumber;
        if (flux0<epsilon2) return epsilon2;
        if (flux0!=flux0) {
            cout    << "Warning flux0 ep; E,T,R = " << energy << " " << theta << " " << R << endl;
            outfile << "Warning flux0 ep; E,T,R = " << energy << " " << theta << " " << R << endl;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.; // Protect against nans
        }
        return flux0;
    } else if (mode==1) { // Return derivative with respect to R
        double dfluxdR = -flux0 * thisp1*thisp2*pow(R,thisp2-1.);  
        if (dfluxdR!=dfluxdR) {
            cout    << "Warning dfluxdR ep" << endl; 
            outfile << "Warning dfluxdR ep" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.;
        }
        // Add derivative of correction factor
        // -----------------------------------
        // dfluxdR = dfluxdR * FC + flux0*FluxCorr(R,1,1);
        return dfluxdR; 
    } else if (mode==2) { // Return derivative with respect to energy

        // Interpolate dthisp0de, dthisp2de - Handle boundaries first
        double dthisp0de, dthisp1de, dthisp2de;
        if (iehig==100) {
            if (ithig==100) { // Upper edge of grid
                dthisp0de = dthisp0de_ep[99][99];
                dthisp1de = dthisp1de_ep[99][99];
                dthisp2de = dthisp2de_ep[99][99];
            } else { // Move on energy edge interpolating in theta
                dthisp0de = dthisp0de_ep[99][itlow]+ dt*(dthisp0de_ep[99][ithig]-dthisp0de_ep[99][itlow]);
                dthisp1de = dthisp1de_ep[99][itlow]+ dt*(dthisp1de_ep[99][ithig]-dthisp1de_ep[99][itlow]);
                dthisp2de = dthisp2de_ep[99][itlow]+ dt*(dthisp2de_ep[99][ithig]-dthisp2de_ep[99][itlow]);
            }
        } else if (ithig==100) { // Theta edge, interpolate in energy
            dthisp0de = dthisp0de_ep[ielow][99]+ de*(dthisp0de_ep[iehig][99]-dthisp0de_ep[ielow][99]);
            dthisp1de = dthisp1de_ep[ielow][99]+ de*(dthisp1de_ep[iehig][99]-dthisp1de_ep[ielow][99]);
            dthisp2de = dthisp2de_ep[ielow][99]+ de*(dthisp2de_ep[iehig][99]-dthisp2de_ep[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0de_ep[ielow][itlow]+ dt*(dthisp0de_ep[ielow][ithig]-dthisp0de_ep[ielow][itlow]);
            double hige0 = dthisp0de_ep[iehig][itlow]+ dt*(dthisp0de_ep[iehig][ithig]-dthisp0de_ep[iehig][itlow]);
            dthisp0de    = lowe0 + de*(hige0-lowe0);
            double lowe1 = dthisp1de_ep[ielow][itlow]+ dt*(dthisp1de_ep[ielow][ithig]-dthisp1de_ep[ielow][itlow]);
            double hige1 = dthisp1de_ep[iehig][itlow]+ dt*(dthisp1de_ep[iehig][ithig]-dthisp1de_ep[iehig][itlow]);
            dthisp1de    = lowe1 + de*(hige1-lowe1);
            double lowe2 = dthisp2de_ep[ielow][itlow]+ dt*(dthisp2de_ep[ielow][ithig]-dthisp2de_ep[ielow][itlow]);
            double hige2 = dthisp2de_ep[iehig][itlow]+ dt*(dthisp2de_ep[iehig][ithig]-dthisp2de_ep[iehig][itlow]);
            dthisp2de    = lowe2 + de*(hige2-lowe2);
        }

        // Note, this is tricky: we are deriving A*exp(f(g(e))) over de.
        // This is like dh(g(f(e)))/de = h'(g(f))g'(f)f'(e),
        // So we get A*exp()*f(g(e))*f'(g(e))*g'(e)
        // That is A*exp()*(-p1*R^p2)*logR*dp2de = flux * R^p2 logR dp2de
        // --------------------------------------------------------------
        double dfluxde = flux0*(1./thisp0*dthisp0de -pow(R,thisp2)*dthisp1de -thisp1*pow(R,thisp2)*log(R)*dthisp2de);
        if (dfluxde!=dfluxde) {
            cout    << "Warning dfluxde ep; R = " << R << " thisp0, p2 = " << thisp0 << " " << thisp2 << " " << dthisp0de << " " << dthisp2de << endl;  
            outfile << "Warning dfluxde ep; R = " << R << " thisp0, p2 = " << thisp0 << " " << thisp2 << " " << dthisp0de << " " << dthisp2de << endl;  
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.;
        }
        // return dfluxde*FC; // Account for tank size
        return dfluxde;
    } else if (mode==3) { // Return derivative with respect to theta

        // Interpolate dthisp0dth, dthisp2dth - Handle boundaries first
        double dthisp0dth, dthisp1dth, dthisp2dth;
        if (iehig==100) {
            if (ithig==100) { // Upper edge of grid
                dthisp0dth = dthisp0dth_ep[99][99];
                dthisp1dth = dthisp1dth_ep[99][99];
                dthisp2dth = dthisp2dth_ep[99][99];
            } else { // Move on energy edge interpolating in theta
                dthisp0dth = dthisp0dth_ep[99][itlow] + dt*(dthisp0dth_ep[99][ithig]-dthisp0dth_ep[99][itlow]);
                dthisp1dth = dthisp1dth_ep[99][itlow] + dt*(dthisp1dth_ep[99][ithig]-dthisp1dth_ep[99][itlow]);
                dthisp2dth = dthisp2dth_ep[99][itlow] + dt*(dthisp2dth_ep[99][ithig]-dthisp2dth_ep[99][itlow]);
            }
        } else if (ithig==100) { // Theta edge, interpolate in energy
            dthisp0dth = dthisp0dth_ep[ielow][99] + de*(dthisp0dth_ep[iehig][99]-dthisp0dth_ep[ielow][99]);
            dthisp1dth = dthisp1dth_ep[ielow][99] + de*(dthisp1dth_ep[iehig][99]-dthisp1dth_ep[ielow][99]);
            dthisp2dth = dthisp2dth_ep[ielow][99] + de*(dthisp2dth_ep[iehig][99]-dthisp2dth_ep[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0dth_ep[ielow][itlow] + dt*(dthisp0dth_ep[ielow][ithig]-dthisp0dth_ep[ielow][itlow]);
            double hige0 = dthisp0dth_ep[iehig][itlow] + dt*(dthisp0dth_ep[iehig][ithig]-dthisp0dth_ep[iehig][itlow]);
            dthisp0dth   = lowe0 + de*(hige0-lowe0);
            double lowe1 = dthisp1dth_ep[ielow][itlow] + dt*(dthisp1dth_ep[ielow][ithig]-dthisp1dth_ep[ielow][itlow]);
            double hige1 = dthisp1dth_ep[iehig][itlow] + dt*(dthisp1dth_ep[iehig][ithig]-dthisp1dth_ep[iehig][itlow]);
            dthisp1dth   = lowe1 + de*(hige1-lowe1);
            double lowe2 = dthisp2dth_ep[ielow][itlow] + dt*(dthisp2dth_ep[ielow][ithig]-dthisp2dth_ep[ielow][itlow]);
            double hige2 = dthisp2dth_ep[iehig][itlow] + dt*(dthisp2dth_ep[iehig][ithig]-dthisp2dth_ep[iehig][itlow]);
            dthisp2dth   = lowe2 + de*(hige2-lowe2);
        }
        // Compute derivative. Careful, it's tricky too: We need to account for the dependence
        // of p0,p1,p2 on theta, but also of the dependence of R itself on theta!
        // -----------------------------------------------------------------------------------
        double dfluxdth   = flux0*(1./thisp0*dthisp0dth 
                                  -thisp1*thisp2*pow(R,thisp2-1.)*dRdTh 
                                  -thisp1*pow(R,thisp2)*log(R)*dthisp2dth
                                  -pow(R,thisp2)*dthisp1dth);
        //double dfluxdth   = flux*(1./thisp0*dthisp0dth 
        //                          -thisp1*thisp2*pow(R,thisp2-1.)*dRdTh 
        //                          -pow(R,thisp2)*log(R)*dthisp2dth);
        if (dfluxdth!=dfluxdth) {
            cout    << "Warning dfluxdth ep" << endl; 
            outfile << "Warning dfluxdth ep" << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings2++;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.;
        } 
        //return dfluxdth*FC; // Account for tank size
        return dfluxdth;
    } else {    
        return 0.;
    }
}

// This function defines a layout which draws the word "MODE" on the ground with the detector positions
// ----------------------------------------------------------------------------------------------------
void DrawMODE(int idfirst, double x_lr, double y_lr, double xystep, double xstep, double ystep) {

    Nunits += 132;
    // M:
    for (int id=0; id<10; id++) {
        x[idfirst+id] = XDoffset + x_lr;
        y[idfirst+id] = YDoffset + y_lr+id*xystep;
    }
    for (int id=0; id<7; id++) {
        x[idfirst+id+10] = XDoffset + x_lr+id*xstep;
        y[idfirst+id+10] = YDoffset + y_lr+10*xystep-id*ystep;
    }
    for (int id=0; id<7; id++) {
        x[idfirst+id+17] = XDoffset + x_lr+7*xstep+id*xstep;
        y[idfirst+id+17] = YDoffset + y_lr+10*xystep-7*ystep+id*ystep;
    }
    for (int id=0; id<11; id++) {
        x[idfirst+id+24] = XDoffset + x_lr+14*xstep;
        y[idfirst+id+24] = YDoffset + y_lr+10*xystep-id*xystep;
    }
    // O:
    for (int id=0; id<10; id++) {
        x[idfirst+id+35] = XDoffset + x_lr+14*xstep+3*xystep;
        y[idfirst+id+35] = YDoffset + y_lr+id*xystep;
    }
    for (int id=0; id<7; id++) {
        x[idfirst+id+45] = XDoffset + x_lr+14*xstep+3*xystep+id*xystep;
        y[idfirst+id+45] = YDoffset + y_lr+10*xystep;
    }
    for (int id=0; id<10; id++) {
        x[idfirst+id+52] = XDoffset + x_lr+14*xstep+10*xystep;
        y[idfirst+id+52] = YDoffset + y_lr+10*xystep-id*xystep;
    }
    for (int id=0; id<7; id++) {
        x[idfirst+id+62] = XDoffset + x_lr+14*xstep+10*xystep-id*xystep;
        y[idfirst+id+62] = YDoffset + y_lr;
    }
    // D:
    for (int id=0; id<10; id++) {
        x[idfirst+id+69] = XDoffset + x_lr+14*xstep+13*xystep;
        y[idfirst+id+69] = YDoffset + y_lr+id*xystep;
    }
    for (int id=0; id<2; id++) {
        x[idfirst+id+79] = XDoffset + x_lr+14*xstep+13*xystep+id*xystep;
        y[idfirst+id+79] = YDoffset + y_lr+10*xystep;
    }
    for (int id=0; id<6; id++) {
        x[idfirst+id+81] = XDoffset + x_lr+14*xstep+15*xystep+id*xstep;
        y[idfirst+id+81] = YDoffset + y_lr+10*xystep-id*ystep;
    }
    for (int id=0; id<3; id++) {
        x[idfirst+id+87] = XDoffset + x_lr+19*xstep+15*xystep;
        y[idfirst+id+87] = YDoffset + y_lr+10*xystep-5*ystep-id*xystep;
    }
    for (int id=0; id<6; id++) {
        x[idfirst+id+90] = XDoffset + x_lr+19*xstep+15*xystep-id*xstep;
        y[idfirst+id+90] = YDoffset + y_lr+7*xystep-5*ystep-id*ystep;
    }
    for (int id=0; id<2; id++) {
        x[idfirst+id+96] = XDoffset + x_lr+13*xstep+15*xystep-id*xystep;
        y[idfirst+id+96] = YDoffset + y_lr;
    }
    // E:
    for (int id=0; id<10; id++) {
        x[idfirst+id+98] = XDoffset + x_lr+19*xstep+18*xystep;
        y[idfirst+id+98] = YDoffset + y_lr+id*xystep;
    }
    for (int id=0; id<8; id++) {
        x[idfirst+id+108] = XDoffset + x_lr+19*xstep+18*xystep+id*xystep;
        y[idfirst+id+108] = YDoffset + y_lr;
        x[idfirst+id+116] = XDoffset + x_lr+19*xstep+18*xystep+id*xystep;
        y[idfirst+id+116] = YDoffset + y_lr+5*xystep;
        x[idfirst+id+124] = XDoffset + x_lr+19*xstep+18*xystep+id*xystep;
        y[idfirst+id+124] = YDoffset + y_lr+10*xystep;
    }
}

// One problem when using aveRdet+Rslack as a boundary for generation of showers is that when the array becomes spread out,
// the average radius plus Rslack can not be enough to generate showers far away from detectors. We correct this by adding
// one sigma to it below.
// ------------------------------------------------------------------------------------------------------------------------
void Rspan () {
    double aveR  = 0.;
    double aveR2 = 0.;
    double maxR  = 0.;
    for (int id=0; id<Nunits; id++) {
        double thisR = sqrt(x[id]*x[id]+y[id]*y[id]);
        if (thisR>maxR) maxR = thisR;
        aveR  += thisR;
        aveR2 += thisR*thisR;
    }
    aveR  = aveR/Nunits;
    aveR2 = aveR2/Nunits;
    ArrayRspan[1] = aveR;
    ArrayRspan[2] = aveR2-aveR*aveR;
    if (ArrayRspan[2]>0.) ArrayRspan[2] = sqrt(ArrayRspan[2]); 
    ArrayRspan[0] = ArrayRspan[1] + 2.*ArrayRspan[2]; // Modify average to be average+2 sigma
    return;
}

// Define the current geometry by updating detector positions
// ----------------------------------------------------------
void DefineLayout () {

    // We create a grid of detector positions.
    // We pave the xy space with a spiral from (0,0):
    // one step up, one right, two down, two left, three up, 
    // three right, four down, four left, etcetera.
    // The parameter SpacingStep widens the step progressively
    // from d to larger values. This allows to study layouts
    // with different density variations from center to periphery;
    // - d = spacing at start of spiral
    // - SpacingStep = rate of increase of spacing
    // - shape = shape of layout:
    //   0       = hexagonal, 
    //   1       = taxi
    //   2       = spiral 
    //   3       = circles 
    //   4       = random box 
    //   5       = word layout 
    //   6       = rectangle extended along y
    //   7       = four circles in a square
    //   8       = circular annulus, empty
    //   9       = double circular annulus
    //   10      = three annuli
    //   11      = rectangle extended along x
    //   12      = hexagonal pattern
    //   101-115 = SWGO original array proposals (also redefines Nunits and TankArea)
    // ------------------------------------------------------------------------------

    // Define units as initially members of multiplets (until ForbiddenRegion decouples them)
    // --------------------------------------------------------------------------------------
    if (CommonMode>1) {
        for (int id=0; id<Nunits; id++) {
            InMultiplet[id] = true;
        }
    }
    // Define mask for fixed units
    // ---------------------------
    for (int id=0; id<Nunits; id++) {
        keep_fixed[id] = false;
    }
    Nactiveunits = Nunits;
    // Work in progress here: try this
    // -------------------------------
    // for (int id=0; id<12; id++) { // ok for 3 3 with 60 units, move only outer ring
    //     keep_fixed[id] = true;
    // } 
    // Nactiveunits = 36;
    if (scanU) Nactiveunits = 1;
    
    // Save for special situations below, the first detector is at 0,0
    // ---------------------------------------------------------------
    x[0] = XDoffset;
    y[0] = YDoffset;
    int id = 1;
    if (shape==0) { // Hexagonal grid
        double deltau = DetectorSpacing;
        double deltav = DetectorSpacing;
        double deltaz = DetectorSpacing;
        int nstepsu = 1;
        int nstepsv = 1;
        int nstepsz = 1;
        double cos30 = sqrt(3.)/2.;
        double sin30 = 0.5;
        int parity = 1.;
        do {
            for (int is=0; is<nstepsu && id<Nunits; is++) {
                x[id] = x[id-1];
                y[id] = y[id-1] + deltau;
                id++;
            }
            deltau = -deltau;
            for (int is=0; is<nstepsv && id<Nunits; is++) {
                x[id] = x[id-1] + deltav*cos30;
                y[id] = y[id-1] + deltav*sin30;
                id++;
            }
            deltav = -deltav;
            for (int is=0; is<nstepsz && id<Nunits; is++) {
                x[id] = x[id-1] + deltaz*cos30;
                y[id] = y[id-1] - deltaz*sin30;
                id++;
            }
            deltaz = -deltaz;
            if (parity==-1) {
                nstepsv++;
            } else {
                nstepsu++;
                nstepsv++;
                nstepsz++;
            }
            parity *= -1;

            // After half cycle we increase the steps size
            // -------------------------------------------
            if (deltau>0) {
                deltau = deltau + SpacingStep;
            } else {
                deltau = deltau - SpacingStep;
            }
            if (deltav>0) {
                deltav = deltav + SpacingStep;
            } else {
                deltav = deltav - SpacingStep;
            }
            if (deltaz>0) {
                deltaz = deltaz + SpacingStep;
            } else {
                deltaz = deltaz - SpacingStep;
            }

        } while (id<Nunits); 
    } else if (shape==1) { // Square grid
        int n_steps = 1;
        double deltax = DetectorSpacing;
        double deltay = DetectorSpacing;
        do {
            for (int is=0; is<n_steps && id<Nunits; is++) {
                x[id] = x[id-1];
                y[id] = y[id-1] + deltay;
                //if (debug) cout << "id = " << id << " x,y = " << x[id] << "," <<  y[id] << endl;
                id++;
            }
            deltay = -deltay;
            for (int is=0; is<n_steps && id<Nunits; is++) {
                x[id] = x[id-1] + deltax;
                y[id] = y[id-1];
                //if (debug) cout << "id = " << id << " x,y = " << x[id] << "," <<  y[id] << endl;
                id++;
            }
            deltax = -deltax;
            n_steps++;
            if (deltax>0) {
                deltax = deltax + SpacingStep;
            } else {
                deltax = deltax - SpacingStep;
            }
            if (deltay>0) {
                deltay = deltay + SpacingStep;
            } else {
                deltay = deltay - SpacingStep;
            }
        } while (id<Nunits); 
    } else if (shape==2) { // Smooth spiral
        double delta = DetectorSpacing;
        double angle0 = 1.; // This better not be a submultiple of 2*pi if spiral_red is close to 1
        double angle = angle0;
        do {
            x[id] = x[id-1] + delta*cos(angle);
            y[id] = y[id-1] + delta*sin(angle);
            id++;
            angle0 = angle0*spiral_reduction;
            angle += angle0;
            delta = delta*SpacingStep; // Step_increase;
            //if (debug) cout << id << " " << angle << " " << delta << " " << cos(angle) << " " << sin(angle) << " " << x[id] << " " << y[id] << endl;
        } while (id<Nunits); 
    } else if (shape==3) {
        int unitspersector = Nunits;
        if (CommonMode>=2) {
            id=0; // Bypass setting central detector at 0,0
            unitspersector = Nunits/multiplicity;
        } 
        int n0 = 6;
        if (multiplicity>6) n0 = multiplicity;
        double r = DetectorSpacing; // First ring
        int n = n0;
        do {
            for (int ith=0; ith<n/multiplicity && id<unitspersector; ith++) {
                double theta = ith*twopi/n;
                x[id] = XDoffset + r*cos(theta);
                y[id] = YDoffset + r*sin(theta);
                // cout << "n = " << n << " ith = " << " r = " << r << " theta = " << theta << " id = " << id << " x,y = " << x[id] << "," << y[id] << endl;
                id++;
            }
            r = r + SpacingStep; // Next ring is separated by a spacingstep
            n = n + n0;
        } while (id<unitspersector);

        // If CommonMode>=2, we repeat the pattern CommonMode times at 360/mult degree spacing
        // -----------------------------------------------------------------------------------
        if (CommonMode>=2) {
            for (int id=0; id<unitspersector; id++) {
                double r = 0.;
                r = pow(x[id],2.)+pow(y[id],2.);
                if (r>0.) r = sqrt(r);
                double phi = PhiFromXY (x[id],y[id]);
                for (int itr=1; itr<multiplicity; itr++) {
                    x[id+itr*unitspersector] = XDoffset + r*cos(phi+itr*twopi/multiplicity); 
                    y[id+itr*unitspersector] = YDoffset + r*sin(phi+itr*twopi/multiplicity); 
                }
            }
        }
    } else if (shape==4) { // Random 2D box distribution, with triangular symmetry
        id = 0; // Bypass first element at 0,0
        for (int id=0; id<Nunits/multiplicity; id++) {
            double halfspan = DetectorSpacing*sqrt(Nunits);
            double r;
            do {
                x[id] = XDoffset + myRNG->Uniform(-halfspan,halfspan);
                y[id] = YDoffset + myRNG->Uniform(-halfspan,halfspan);
                r = pow(x[id],2.)+pow(y[id],2.);
            } while (r<=0.);
            r = sqrt(r);
            double phi = PhiFromXY (x[id],y[id]);
            for (int itr=1; itr<multiplicity; itr++) {
                x[id+itr*Nunits/multiplicity]   = XDoffset + r*cos(phi+2.*pi*itr/multiplicity);
                y[id+itr*Nunits/multiplicity]   = XDoffset + r*sin(phi+2.*pi*itr/multiplicity);
            }
        }
    } else if (shape==5) { // Word layout
        int idfirst   = 0;
        double x_lr   = -410;
        double y_lr   = -50;
        double xystep = 10;
        double xstep  = 7;
        double ystep  = 7;
        Nunits = 0;
        DrawMODE(idfirst,x_lr,y_lr,xystep,xstep,ystep);
        idfirst   = 132;
        x_lr   = 40;
        y_lr   = -50;
        xystep = 10;
        xstep  = 7;
        ystep  = 7;
        DrawMODE(idfirst,x_lr,y_lr,xystep,xstep,ystep);
        //    Nunits = 132;
        idfirst = 264;
        x_lr    = -200;
        y_lr    = 250;
        xystep  = 10;
        xstep   = 7;
        ystep   = 7;
        DrawMODE(idfirst,x_lr,y_lr,xystep,xstep,ystep);
        idfirst = 396;
        // Nunits = 396;
        x_lr    = -200;
        y_lr    = -350;
        xystep  = 10;
        xstep   = 7;
        ystep   = 7;
        DrawMODE(idfirst,x_lr,y_lr,xystep,xstep,ystep);
        Nunits  = 528;
    } else if (shape==6) { // Rectangle
        for (int id=0; id<Nunits; id++) {
            x[id] = XDoffset -0.5*DetectorSpacing+DetectorSpacing*(id%2);
            y[id] = YDoffset -(Nunits-2)*0.25*DetectorSpacing+DetectorSpacing*int(id/2);
        }
    } else if (shape==7) { // Four circular densities in a square 
        double Qsize = sqrt(Nunits)*DetectorSpacing; 
        double XVoffset, YVoffset;
        id = 0;
        for (int ivertex=0; ivertex<4; ivertex++) {
            XVoffset = -0.5*Qsize + (ivertex%2)*Qsize;
            YVoffset = -0.5*Qsize + (ivertex/2)*Qsize;
            double r = DetectorSpacing;
            do {
                double n = 6.*r/DetectorSpacing;
                for (int ith=0; ith<n && id<(ivertex+1)*Nunits/4; ith++) {
                    double theta = ith*twopi/n;
                    x[id] = XDoffset + r*cos(theta) + XVoffset;
                    y[id] = YDoffset + r*sin(theta) + YVoffset;
                    id++;
                }
                r = r + SpacingStep;
            } while (id<(ivertex+1)*Nunits/4);
        }
    } else if (shape==8) { // One annulus
        double r = DetectorSpacing*Nunits/(2*pi);
        double n = Nunits;
        for (int id=0; id<n/multiplicity; id++) {
            double phi = id*twopi/n;
            x[id] = XDoffset + r*cos(phi);
            y[id] = YDoffset + r*sin(phi);
            for (int itr=1; itr<multiplicity; itr++) {
                x[id+itr*Nunits/multiplicity]   = XDoffset + r*cos(phi+2.*pi*itr/multiplicity);
                y[id+itr*Nunits/multiplicity]   = XDoffset + r*sin(phi+2.*pi*itr/multiplicity);
            }
        }
    } else if (shape==9) { // Two annuli
        int unitspersector = Nunits;
        if (CommonMode>=2) {
            id=0; // Bypass setting central detector at 0,0
            unitspersector = Nunits/multiplicity;
        } 
        double r[2];
        r[0] = DetectorSpacing*Nunits/(6.*pi);
        r[1] = DetectorSpacing*Nunits/(3.*pi);
        double n[2];
        n[0] = Nunits/3;
        n[1] = Nunits-n[0];
        int ir = 0;
        do {
            double n = 6.*r[ir]/DetectorSpacing;
            for (int ith=0; ith<n/multiplicity && id<unitspersector; ith++) {
                double theta = ith*twopi/n;
                x[id] = XDoffset + r[ir]*cos(theta);
                y[id] = YDoffset + r[ir]*sin(theta);
                id++;
            }
            ir++;
        } while (id<unitspersector && ir<2);

        // If CommonMode>=2, we repeat the pattern CommonMode times at 360/mult degree spacing
        // -----------------------------------------------------------------------------------
        if (CommonMode>=2) {
            for (int id=0; id<unitspersector; id++) {
                double r = 0.;
                r = pow(x[id],2.)+pow(y[id],2.);
                if (r>0.) r = sqrt(r);
                double phi = PhiFromXY (x[id],y[id]);
                for (int itr=1; itr<multiplicity; itr++) {
                    x[id+itr*unitspersector] = XDoffset + r*cos(phi+itr*twopi/multiplicity); 
                    y[id+itr*unitspersector] = YDoffset + r*sin(phi+itr*twopi/multiplicity); 
                }
            }
        }
    } else if (shape==10) { // Three annuli
        if (Nunits%(3*multiplicity)!=0) {
            Nunits = Nunits + 3*multiplicity - Nunits%(3*multiplicity);
            cout << "     For this config Nunits has to be multiple of " << 3*multiplicity << ". Changing it to next integer, " << Nunits << endl;
        }
        double r[3];
        r[0] = DetectorSpacing*Nunits/(6.*pi);
        r[1] = DetectorSpacing*Nunits/(3.*pi);
        r[2] = DetectorSpacing*Nunits/(2.*pi);
        double n[3];
        n[0] = Nunits/6;
        n[1] = Nunits/3;
        n[2] = Nunits/2;
        int done = 0;
        for (int ir=0; ir<3; ir++) {
            for (int id=0; id<n[ir]/multiplicity; id++) {
                for (int itr=0; itr<multiplicity; itr++) {
                    double phi = twopi*(1.*id/n[ir] + 1.*itr/multiplicity);
                    int index = done + id + itr*Nunits/multiplicity;
                    x[index]   = XDoffset + r[ir]*cos(phi);
                    y[index]   = XDoffset + r[ir]*sin(phi);
                    //cout << "ir= " << ir << " id= " << id << " ind= " << index << " r= " << r[ir] << " phi= " << phi << " x= " << x[index] << " y= " << y[index] << endl;  
                }
            } 
            done += n[ir]/multiplicity;
        }
    } else if (shape==11) { // Rectangle extended along x
        for (int id=0; id<Nunits; id++) {
            y[id] = YDoffset -0.5*DetectorSpacing+DetectorSpacing*(id%2);
            x[id] = XDoffset -(Nunits-2)*0.25*DetectorSpacing+DetectorSpacing*int(id/2);
        }
    } else if (shape==12) { 
        // This choice generates three zones of different fill factors, mimicking the A7 layout. 
        // Note that this requires resetting to zero the slack between tanks set in D2min() 
        // -------------------------------------------------------------------------------------
        // The first detector is already set at 0,0 above, and id=1. We remove it to have the 
        // first packed circle contain 120 units once we also remove the corners of the hexagon,
        // because in the A7 array there are 2335/2394/1842 units, and we want an array of 330,
        // so we have the three regions containing 120/120/90 units which nicely fits the pattern
        // of 6 rings of 6+12+18+24+30+36 in the inner circle, with minimal spacing between the
        // tanks and total radius of 114m; 5 rings at radii 200,300,400,500,600m with 12+18+24+30+36
        // units; and 4 rings at radii 750,900,1050,1200m with 12+18+24+30 units.
        // ----------------------------------------------------------------------------------------- 
        id = 0;
        int n0 = 6; // how many more dets for each new ring
        double r = DetectorSpacing; // First ring
        int n = 6;  // first ring has 6
        do {
            for (int ith=0; ith<n && id<120; ith++) {
                double theta = ith*twopi/n;
                x[id] = XDoffset + r*cos(theta);
                y[id] = YDoffset + r*sin(theta);
                cout << "n = " << n << " ith = " << " r = " << r << " theta = " << theta << " id = " << id << " x,y = " << x[id] << "," << y[id] << endl;
                id++;
                // We remove the six vertices of the outer ring
                // These have id=90,96,102,108,114,120.
                // --------------------------------------------
                if (id==90||id==96||id==102||id==108||id==114||id==120) ith++;
            }
            r = r + SpacingStep; // Next ring is separated by a spacingstep
            n = n + n0;
        } while (id<120);        
        // For the second ring, we arrange again 120 detectors in 5 rings of 12+18+24+30+36 units
        // starting at a radius of 200m and with 100m increments to 600m
        // --------------------------------------------------------------------------------------
        n0 = 6;   // how many more per each new ring
        r = 200.; // First ring of this set of five
        n = 12;   // Start with 12 elements
        do {
            for (int ith=0; ith<n && id<240; ith++) {
                double theta = ith*twopi/n;
                x[id] = XDoffset + r*cos(theta);
                y[id] = YDoffset + r*sin(theta);
                cout << "n = " << n << " ith = " << " r = " << r << " theta = " << theta << " id = " << id << " x,y = " << x[id] << "," << y[id] << endl;
                id++;
            }
            r = r + 100; // Next ring is separated by 100m
            n = n + n0;
        } while (id<240);        
        // For the third and outermost ring, we arrange 90 detectors in 4 rings of 12+18+24+30 units
        // starting at a radius of 200m and with 100m increments to 600m
        // --------------------------------------------------------------------------------------
        n0 = 3;   // how many more per each new ring
        r = 750.; // First ring of this set of five
        n = 18;   // Start with 18 elements
        do {
            for (int ith=0; ith<n && id<330; ith++) {
                double theta = ith*twopi/n;
                x[id] = XDoffset + r*cos(theta);
                y[id] = YDoffset + r*sin(theta);
                cout << "n = " << n << " ith = " << " r = " << r << " theta = " << theta << " id = " << id << " x,y = " << x[id] << "," << y[id] << endl;
                id++;
            }
            r = r + 150; // Next ring is separated by 100m
            n = n + n0;
        } while (id<330);        
    } else if (shape>100 && shape<115) {
        // SWGO layouts. Coding is the following:
        // 101 == A1, 6589 entries
        // 102 == A2, 6631 entries
        // 103 == A3, 6823
        // 104 == A4, 6625
        // 105 == A5, 6541
        // 106 == A6, 6637
        // 107 == A7, 6571
        // 108 == B1, 4849
        // 109 == C1, 8371
        // 110 == D1, 3805
        // 111 == E1, 5461
        // 112 == E4, 5455
        // 113 == F1, 4681 
        // 114 == D8, 3763 - note, different format in files (positions in cm, no number of tank)

        // Set detector area to nominal value (we do not deal with aggregates of tanks here)
        // ---------------------------------------------------------------------------------
        TankArea = TankRadius*TankRadius*pi;
        if (shape==101) Nunits = 6589;
        if (shape==102) Nunits = 6631;
        if (shape==103) Nunits = 6823;
        if (shape==104) Nunits = 6625;
        if (shape==105) Nunits = 6541;
        if (shape==106) Nunits = 6636;
        if (shape==107) Nunits = 6571;
        if (shape==108) Nunits = 4849;
        if (shape==109) Nunits = 8371;
        if (shape==110) Nunits = 3805;
        if (shape==111) Nunits = 5461;
        if (shape==112) Nunits = 5455;
        if (shape==113) Nunits = 4681;
        if (shape==114) Nunits = 3763;

#ifdef STANDALONE
        string detPath  = GlobalPath + "Dets/"; // "/lustre/cmswork/dorigo/swgo/MT/Dets/";
#endif
#ifdef UBUNTU
        string detPath = GlobalPath + "Dets/";
#endif 
#ifdef INROOT
        string detPath  = "./SWGO/dets/";
#endif
        ifstream detfile;
        char num[40];
        sprintf (num,"Layout_%d", shape);
        string detPositions = detPath  + num + ".txt";
        detfile.open(detPositions);
        double e;
        
        if (shape<114) {
            for (int id=0; id<Nunits; id++) {
                detfile >> e;
                if (e!=id) {
                    cout << "Problem reading layout file" << endl;
                    return;
                }
                detfile >> e;
                x[id] = e;
                detfile >> e;
                y[id] = e;
                // cout << "   Unit # " << id << ": x,y = " << x[id] << " " << y[id] << endl;
            }
        } else { // For D8, we do not have det numbers, and positions are in cm
            for (int id=0; id<Nunits; id++) {
                detfile >> e;
                x[id] = e/100.;
                detfile >> e;
                y[id] = e/100.;
            }
        }
        detfile.close();
        // cout << endl;        

        cout << endl;
        cout << "     ---------- Layout read in from config # " << shape << " ----------" << endl << endl;
        outfile << endl;
        outfile << "     ---------- Layout read in from config. # " << shape << " ----------" << endl << endl;
        /* 
        for (int id=0; id<Nunits; id++) {
            outfile << "    Unit # " << id << ": x,y = " << x[id] << " " << y[id] << endl;
        }
        outfile << "     -------------------------------------------------------" << endl << endl;
        outfile << endl;
        */

    } // end if shape

    // Define span x and y of generated showers to illuminate initial layout
    // ---------------------------------------------------------------------
    Rspan(); // Fills ArrayRspan[0-2] values - we need to initialize them even if we do not use them for totalrspan
    TotalRspan = ArrayRspan[0]+Rslack;

    // Enforce forbidden region
    // ------------------------
    if (VoidRegion) {
        //IncludeOnlyFR = false;
        //FRpar[0] = 0.5;
        //FRpar[1] = 200.;
        //ForbiddenRegion (1,0,0); // in y region below line y=0.5x+200
        //IncludeOnlyFR = false;
        //FRpar[0] = -100.;
        //FRpar[1] = 300.;
        //ForbiddenRegion (2,0,0); // in x region [-100,300] 
        // ----------------------------------------------------
        // Circle
        //IncludeOnlyFR = true;
        //FRpar[0] = 0.;
        //FRpar[1] = 0.;
        //FRpar[2] = 1200.;
        //ForbiddenRegion(0,0,0); // within a 1200 m circle
        // Pampalabola: triangle at positions
        // A = (-3499.,2000.) B = (1800,2000) C = (1800,-3508). Tangent of BAC is -0.804735, intercept -816
        // ---------------------
        IncludeOnlyFR = false;
        FRpar[0] = -3499.;
        FRpar[1] = 1800.;
        ForbiddenRegion(2,0,0); // Vertical semiplane
        IncludeOnlyFR = true;
        FRpar[0] = -tan(0.804735);
        FRpar[1] = -1637.;
        ForbiddenRegion(1,0,0); // Non vertical semiplane defined by y=mx+q
        IncludeOnlyFR = false;
        FRpar[0] = 0.;
        FRpar[1] = 2000.;
        ForbiddenRegion(1,0,0); // Non vertical semiplane defined by y=mx+q
    }
    if (shape<101) ResolveOverlaps (); // For now we do not keep this on when Ntanks is very large (14 default arrays)

    return;
}

// Function that saves the layout data to file
// -------------------------------------------
void SaveLayout () {
#ifdef STANDALONE
    string detPath  = "/lustre/cmswork/dorigo/swgo/MT/Dets/";
#endif
#ifdef UBUNTU
    string detPath  = "/home/tommaso/Work/swgo/MT/Dets/";
#endif
#ifdef INROOT
    string detPath  = "./SWGO/Detectors/";
#endif
    ofstream finaldetfile;
    std::stringstream sstr;
    char num[60];
    // Note, here we take the same indfile of the RunDetails output file, to write out the detector geometry.
    // See note at beginning of main routine concerning indfile (after RunDetails file is defined)
    // ------------------------------------------------------------------------------------------------------
    sprintf (num,"Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile);
    sstr << "Layout_";
    string detPositions = detPath  + sstr.str() + num + ".txt";
    cout << "     Writing to output file " << detPositions << " the detector positions" << endl;
    finaldetfile.open(detPositions);
    for (int id=0; id<Nunits; id++) {
        finaldetfile << x[id] << " " << y[id] << " " << endl;
    }
    // Save size of learning rate for further runs
    // -------------------------------------------
    finaldetfile << 1. + finalx_prevLR << endl;
    finaldetfile.close();
    return;
}

// Dump warnings if terminating
// ----------------------------
void TerminateAbnormally () {
    cout    << "     There were serious warnings, terminating. " << endl;
    cout    << "     Warnings: " << endl;
    cout    << "     1 - " << warnings1 << endl;
    cout    << "     2 - " << warnings2 << endl;
    cout    << "     3 - " << warnings3 << endl;
    cout    << "     4 - " << warnings4 << endl;
    cout    << "     5 - " << warnings5 << endl;
    cout    << "     6 - " << warnings6 << endl;
    cout    << "     6 - " << warnings7 << endl;
    cout    << "--------------------------------------------------------------" << endl;

    // Close dump file
    // ---------------
    outfile << endl;
    outfile << "The program terminated due to warnings. " << endl;
    outfile << "     Warnings: " << endl;
    outfile << "     1 - " << warnings1 << endl;
    outfile << "     2 - " << warnings2 << endl;
    outfile << "     3 - " << warnings3 << endl;
    outfile << "     4 - " << warnings4 << endl;
    outfile << "     5 - " << warnings5 << endl;
    outfile << "     6 - " << warnings6 << endl;
    outfile << "     6 - " << warnings7 << endl;
    outfile << "--------------------------------------------------------------" << endl;
    outfile.close();
    return;
}

// Function that reads the layout data from file
// ---------------------------------------------
void ReadLayout () {
#ifdef STANDALONE
    string detPath  = "/lustre/cmswork/dorigo/swgo/MT/Dets/";
#endif
#ifdef UBUNTU
    string detPath  = "/home/tommaso/Work/swgo/MT/Dets/";
#endif
#ifdef INROOT
    string detPath  = "./SWGO/Detectors/";
#endif
    ifstream detfile;
    std::stringstream sstr;
    char num[100];
    string detPositions;

    // We can either read the InputLayout.txt file (user defined ad hoc) or a previously generated layout 
    // from an optimization run.
    // --------------------------------------------------------------------------------------------------
    if (PredefinedLayout) {
        sprintf (num, "%d_%d", Nunits, PredefinedLayoutID);
        sstr << "InputLayout_";
        detPositions = detPath  + sstr.str() + num + ".txt";
    } else {
        // Determine last file number to read. Since there may be index voids in a sequence of files with same name
        // (e.g. when a job was terminated before writing the output layout), we look for indices up to 100 times
        // larger before determining which file to read.
        // --------------------------------------------------------------------------------------------------------
        int currentfile = -1;
        int lastgood = -1;
        ifstream tmpfile;
        do {
            currentfile++;
            sprintf (num, "Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d", Nbatch, Nunits, startEpoch-Nepochs, startEpoch, shape, currentfile); // NNBB assumes we run jobs of same Nepochs in succession
            std::stringstream tmpstring;
            tmpstring << "Layout_";
            string tmpfilename = detPath + tmpstring.str() + num + ".txt";
            tmpfile.open(tmpfilename);
            if (tmpfile.is_open()) {
                lastgood = currentfile;
                tmpfile.close();
            }
        } while (lastgood>currentfile-100);

        sprintf (num,"Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d", Nbatch, Nunits, startEpoch-Nepochs, startEpoch, shape, lastgood);
        sstr << "Layout_";
        detPositions = detPath  + sstr.str() + num + ".txt";
    }

    // Open file and read positions
    // ----------------------------
    detfile.open(detPositions);
    double e;
    cout    << endl;
    cout    << "     ---------- Layout read in from previous run: ----------" << endl;
    cout    << "                File is " << detPositions << endl;
    cout    << endl;
    outfile << endl;
    outfile << "     ---------- Layout read in from previous run: ----------" << endl << endl;
    outfile << "                File is " << detPositions << endl;
    outfile << endl;
    for (int id=0; id<Nunits; id++) {
        detfile >> e;
        x[id] = e;
        detfile >> e;
        y[id] = e;
        // cout << "     Unit # " << id << ": x,y = " << x[id] << " " << y[id] << endl;
    }
    // Read in the value taken by the learning rate normalization at the last epoch in stored run
    // ------------------------------------------------------------------------------------------
    detfile >> e;
    finalx_prevLR = e;
    cout    << "     Read in the argument of the learning rate to which we got in previous run, " << finalx_prevLR << endl;
    cout    << endl;
    outfile << "     Read in the argument of the learning rate to which we got in previous run, " << finalx_prevLR << endl;
    outfile << endl;
    detfile.close();

    // Define units as initially members of multiplets (until ForbiddenRegion decouples them)
    // --------------------------------------------------------------------------------------
    if (CommonMode>1) {
        for (int id=0; id<Nunits; id++) {
            InMultiplet[id] = true;
        }
    }
    // Define mask for fixed units
    // ---------------------------
    for (int id=0; id<Nunits; id++) {
        keep_fixed[id] = false;
    }
    Nactiveunits = Nunits;


    // If expanding TankNumber to 1 per unit, we need to redefine x,y, Nunits:
    // -----------------------------------------------------------------------
#ifdef EXPANDARRAY
    int Nunits_real = Nunits*TankNumber;  

    // Store coordinates of centers of macro-tanks
    // -------------------------------------------
    for (int id=0; id<Nunits; id++) {
        xprev[id] = x[id];
        yprev[id] = y[id];
    }
    // Arrange single tanks in packed circular structure
    // -------------------------------------------------
    int idnew = 0;
    for (int id=0; id<Nunits; id++) {
        // Place the first unit at the center
        idnew = id*TankNumber;  
        x[idnew] = xprev[id];
        y[idnew] = yprev[id];
        idnew++;
        // Arrange TankNumber units around id-th
        int n0 = 6;
        double r = 0.6+1.91*2.; // First ring
        int n = n0;
        do {
            for (int ith=0; ith<n; ith++) {
                double theta = ith*twopi/n;
                x[idnew] = xprev[id] + r*cos(theta);
                y[idnew] = yprev[id] + r*sin(theta);
                idnew++;
            }
            r = r + 0.6+1.91*2.; // next ring is separated by a spacingstep
            n = n + n0;
        } while (idnew<(id+1)*TankNumber); 
    }
    // Redefine true number of units
    // -----------------------------
    Nunits = Nunits_real; 
    if (Nunits!=idnew) {
        cout    << "     Mismatch in array expansion, " << Nunits << " " << idnew+1 << endl;
        outfile << "     Mismatch in array expansion, " << Nunits << " " << idnew+1 << endl;
        TerminateAbnormally();
    } else {
        cout    << "     Successfully expanded into " << Nunits << " array." << endl;
        outfile << "     Successfully expanded into " << Nunits << " array." << endl;
    }
#endif

    // Define span x and y of generated showers to illuminate initial layout
    // ---------------------------------------------------------------------
    Rspan(); // Fills ArrayRspan[0-2] values
    TotalRspan = ArrayRspan[0]+Rslack;

    return;
}

// This function computes an approximation to the probability that a shower passes the
// condition N(units with >=1 observed particle) >= Ntrigger. See routine CheckProb
// and explanation in comments around computation of Pg, Pp. The function is called
// during GenerateShower to set the value of PActive[is].
// Note: if mode==0 the probability is computed (PActive[is])
// If instead mode == 1 or 2, the derivative is returned of PActive[is]
// with respect to a movement of detector id along x or y, respectively. 
// -----------------------------------------------------------------------------------
double ProbTrigger (double SumProbs, int mode=0, int id=0, int is=0) {
    if (mode==0) {
        double sum = 0.;
        if (SumProbs>Ntrigger+SumProbRange) return 1.; 
        if (SumProbs<Ntrigger-SumProbRange) return 0.; 

        // Evaluate ProbTrigger as a tail prob to see k or more tanks firing, 
        // if the sum of tanks firing probabilities is SumProbs
        // ------------------------------------------------------------------
        for (int k=0; k<Ntrigger; k++) {
            sum += MyPoisson (k,SumProbs); // exp(-SumProbs)*pow(SumProbs,k)/F[k]; // F is factorial
        }
        if (sum!=sum) {
            cout    << "Warning, trouble in ProbTrigger." << endl;
            outfile << "Warning, trouble in ProbTrigger." << endl;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings6++; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 1.;
        }
        double p = 1.-sum;
        if (p<0.) {
            cout    << "WTF Pactive = " << p << endl;
            outfile << "WTF Pactive = " << p << endl;
            return 0.;
        } else if (p>1.) {
            return 1.;
        } else {
            return p;
        }
    } else if (mode==1) { // Compute derivative wrt x[id]

        // Get dPA_dx, dPA_dy
        // To incorporate the contribution of PActive_m, in the calculation of dU/dx, we have made it part of the def of G.
        // This means we have a term to add to the calculations in ThreadFunction2, G/PActive_m * dPActive_m/dxi. 
        // The latter term can be computed as follows (PActive = 1 - sum(1:Ntr-1)):
        //     dPActive_m/dxi = -d/dxi [sum_{j=0}^{Ntr-1}(e^{-Sm}*Sm^j/j!)]
        // with
        //     Sm = Sum_{i=1}^Ndet [1-exp(-lambda_mu^i-lambda_e^i)_m]
        // We get
        //     dPActive_m/dxi = - Sum_{j=0}^{Ntr-1} 1/j! [e^{-Sm}*(Sm^j-j*Sm^(j-1))*dSm/dxi]
        // Now for dSm/dxi we have
        //     dSm/dxi = d/dxi [ Sum_{i=1}^{Ndet} (1-e^(-lambda_mu^i-lambda_e^i)_m))]
        //               = -d/dxi (e^[-lambda_mu^i-lambda_e^i]_m) =
        //               = e^[-lambda_mu^i-lambda_e^i]*(dlambda_mu^i/dxi + dlambda_e^i/dxi)
        // and the latter are computed in the flux routines.
        // Note that the lambdas are the true ones, as we compute PDFs with events that are
        // included in the sum if they pass the trigger, and they do if the true fluxes exceed
        // the threshold of Ntrigger detectors firing.
        // -----------------------------------------------------------------------------------
        // The calculation of the term due to dPActive/dxi is laborious, but we only need
        // to perform it if we are close to threshold, otherwise the derivative contr. is null
        // -----------------------------------------------------------------------------------
        double dPAm_dx = 0.;
        double xt   = TrueX0[is];
        double yt   = TrueY0[is];
        double tt   = TrueTheta[is];
        double pt   = TruePhi[is];
        double et   = TrueE[is];
        double ctt  = cos(tt);
        double Rim  =  EffectiveDistance (x[id],y[id],xt,yt,tt,pt,0);
        double dRdx = -EffectiveDistance (x[id],y[id],xt,yt,tt,pt,1); // Wrt xi, so negative sign
        double lm, le, dlmdR, dledR;
        if (IsGamma[is]) {
            lm    = MFromG (et,tt,Rim,0)*ctt + fluxB_mu; // The second term is the background
            le    = EFromG (et,tt,Rim,0)*ctt + fluxB_e;
            dlmdR = MFromG (et,tt,Rim,1)*ctt;
            dledR = EFromG (et,tt,Rim,1)*ctt;
        } else {
            lm    = MFromP (et,tt,Rim,0)*ctt + fluxB_mu; // The second term is the background
            le    = EFromP (et,tt,Rim,0)*ctt + fluxB_e;
            dlmdR = MFromP (et,tt,Rim,1)*ctt;
            dledR = EFromP (et,tt,Rim,1)*ctt;
        }
        double dlmdx  = dlmdR*dRdx;
        double dledx  = dledR*dRdx;
        double dSm_dx;
        if (TankNumber==1) {
            dSm_dx = exp(-lm-le)*(dlmdx+dledx);
        } else {
            dSm_dx = TankNumber*(dlmdx+dledx)*pow(1.-1./TankNumber,lm+le)*log(1.-1./TankNumber); // exp(-lm-le)*(dlmdx+dledx);
        }
        for (int j=0; j<Ntrigger; j++) {
            if (j>0) {
                dPAm_dx += dSm_dx * (-MyPoisson(j-1,SumProbs)+MyPoisson(j,SumProbs));            
            // cout << "1 j="<< j << " s=" << SumProbs << " " << (-OldPoisson(j-1,SumProbs)+OldPoisson(j,SumProbs)) << " " << (-MyPoisson(j-1,SumProbs)+MyPoisson(j,SumProbs)) << endl; 
            } else {
                dPAm_dx += dSm_dx * MyPoisson(j,SumProbs);  // There is no term j*sm^j-1 if j=0          
            // cout << "2 j=" << j << " s=" << SumProbs << " " << OldPoisson(j,SumProbs) << " " << MyPoisson(j,SumProbs) << endl; 
            }
            // (exp(-SumProbs)*(pow(SumProbs,j)-j*pow(SumProbs,j-1))*dSm_dx)/F[j]; // F[j] is factorial
        }
        if (dPAm_dx!=dPAm_dx) {
            cout    << "Warning, trouble in dPActive/dx calculation." << endl;
            outfile << "Warning, trouble in dPActive/dx calculation." << endl;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings4++; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.;
        }
        return dPAm_dx;
    } else if (mode==2) { // Compute derivative wrt y[id]
        double dPAm_dy = 0.;
        double xt   = TrueX0[is];
        double yt   = TrueY0[is];
        double tt   = TrueTheta[is];
        double pt   = TruePhi[is];
        double et   = TrueE[is];
        double ctt  = cos(tt);
        double Rim  =  EffectiveDistance (x[id],y[id],xt,yt,tt,pt,0);
        double dRdy = -EffectiveDistance (x[id],y[id],xt,yt,tt,pt,2); // wrt yi
        double lm, le, dlmdR, dledR;
        if (IsGamma[is]) {
            lm    = MFromG (et,tt,Rim,0)*ctt + fluxB_mu; 
            le    = EFromG (et,tt,Rim,0)*ctt + fluxB_e;
            dlmdR = MFromG (et,tt,Rim,1)*ctt;
            dledR = EFromG (et,tt,Rim,1)*ctt;
        } else {
            lm    = MFromP (et,tt,Rim,0)*ctt + fluxB_mu; 
            le    = EFromP (et,tt,Rim,0)*ctt + fluxB_e;
            dlmdR = MFromP (et,tt,Rim,1)*ctt;
            dledR = EFromP (et,tt,Rim,1)*ctt;
        }
        double dlmdy  = dlmdR*dRdy;
        double dledy  = dledR*dRdy;
        double dSm_dy;
        if (TankNumber==1) {
            dSm_dy = exp(-lm-le)*(dlmdy+dledy);
        } else {
            dSm_dy = TankNumber*(dlmdy+dledy)*pow(1.-1./TankNumber,lm+le)*log(1.-1./TankNumber); // exp(-lm-le)*(dlmdx+dledx);
        }
        for (int j=0; j<Ntrigger; j++) {
            if (j>0) {
                dPAm_dy += dSm_dy * (-MyPoisson(j-1,SumProbs)+MyPoisson(j,SumProbs));            
            } else {
                dPAm_dy += dSm_dy * MyPoisson(j,SumProbs);  // there is no term j*sm^j-1 if j=0          
            }
            // (exp(-SumProbs)*(pow(SumProbs,j)-j*pow(SumProbs,j-1))*dSm_dx)/F[j]; // F[j] is factorial
        }
        if (dPAm_dy!=dPAm_dy) {
            cout    << "Warning, trouble in dPActive/dy calculation." << endl;
            outfile << "Warning, trouble in dPActive/dy calculation." << endl;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings4++; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            return 0.;
        }
        return dPAm_dy;
    }
    cout    << "Warning, invalid mode in ProbTrigger, " << mode << endl;
    outfile << "Warning, invalid mode in ProbTrigger, " << mode << endl;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings4++; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
    return 0.; // This should not happen
} 

// Get counts and times in detector units
// --------------------------------------
void GetCounts (int is) {

    // We need to redo these initializations, already done at start of GenerateShower,
    // just in case we are coming in here from a replication of the event in findlogLR (when sampleT is on)
    // ----------------------------------------------------------------------------------------------------
    double Ncount  = 0;
    Nunits_npgt0   = 0; // Counts units with positive number of particles detected, for DoF calculation
    SumProbGe1[is] = 0.;
    Active[is]     = false;
    PActive[is]    = 0.;
    float Npexp    = 0.;

    // Loop on units to get counts and times in each
    // ---------------------------------------------
    for (int id=0; id<Nunits; id++) {   
        Nm[id][is] = 0;
        Ne[id][is] = 0;
        Tm[id][is] = 0.;
        Te[id][is] = 0.;
        //Stm[id][is] = sigma_time; // just an initialization. For now these are omitted 
        //Ste[id][is] = sigma_time; // 
        double R = EffectiveDistance (x[id],y[id],TrueX0[is],TrueY0[is],TrueTheta[is],TruePhi[is],0);
        double ct = cos(TrueTheta[is]);
        double m0, e0, mb, eb;
        double fbm = 0.;
        double fbe = 0.;
        float nms, nes, nmb, neb;
        if (IsGamma[is]) {
            m0 = MFromG (TrueE[is],TrueTheta[is],R,0) * ct;
            e0 = EFromG (TrueE[is],TrueTheta[is],R,0) * ct;
        } else {
            m0 = MFromP (TrueE[is],TrueTheta[is],R,0) * ct;
            e0 = EFromP (TrueE[is],TrueTheta[is],R,0) * ct;
        }
        mb = fluxB_mu;
        eb = fluxB_e;
        if (m0+mb>0.) fbm = mb/(m0+mb);
        if (e0+eb>0.) fbe = eb/(e0+eb);
        // We simulate counts by first obtaining the actual number of particles showing up in a tank, 
        // and then smearing that number out by RelResCounts. This means to account for the resolution
        // of the counting of tracks from photoelectron yields.
        // -------------------------------------------------------------------------------------------
        if (SameShowers) {
            // We need separately the number of particles produced by signal and background, as the
            // time distributions of the components are different (Gauss vs Uniform)
            // ------------------------------------------------------------------------------------
            nms = SmearN(m0);
            nes = SmearN(e0);
            nmb = SmearN(mb);
            neb = SmearN(eb);
        } else {
            nms = SmearN(myRNG->Poisson(m0));
            nes = SmearN(myRNG->Poisson(e0));
            nmb = SmearN(myRNG->Poisson(mb));
            neb = SmearN(myRNG->Poisson(eb));
        }
        Nm[id][is] = nms+nmb; // Integer part is taken - these are the observed counts in a macro-tank
        Ne[id][is] = nes+neb; // Integer part is taken - these are the observed counts in a macro-tank
        if (nms+nmb+nes+neb>0.) Nunits_npgt0++; // This is used for N_DoF calculations
        Npexp = nms+nmb+nes+neb; // Float: this is the expectation number of particles seen in total by a macro-tank
        int Nobs = Nm[id][is]+Ne[id][is]; // Integer

        // Each unit in the id loop is a macrotank constituted by TankNumber tanks. The Ntrigger condition
        // applies to the number of tanks Ncount that are seeing a signal, among Nunits*TankNumber units. 
        // So we need to compute this, and we rely on the sum of probabilities SumProbGe1 of seeing particles 
        // that each macrotank has. The calculation of 
        // SumProbGe1 relies on the expected number of tanks with >=1 observed particles, which is obtained
        // by the expected total flux in the macrotank, Npexp
        // -----------------------------------------------------------------------------------------------
        SumProbGe1[is] += TankNumber * (1. - exp(-Npexp/TankNumber)); 
        // The above uses Poisson prob P(0|mu) = exp(-mu), with mu = Npexp/TankNumber, as the prob that one single
        // tank in the macroaggregate sees no particles when Npexp are expected in total in the aggregate.
        // With it we calculate the sum of probabilities of seeing >=1 particle in the considered tanks,
        // which is what we need for the approximate calculations later.
        //
        // While the above is computed using the expected number of particles Npexp, if we want to model
        // a realization of a given number Nm+Ne particles distributed in TN tanks (for the purpose of
        // checking if Ntrigger is satisfied with the generated shower, rather than computing the probability
        // of triggering of the expected flux), we need to use Bernoulli trials, not the Poisson distribution.
        // In this case we have <N_empty> = TN*[(TN-1)/TN)]^(Nm+Ne), so N_signal = TN * {1-[(TN-1)/TN]^(Nm+Ne)}
        // if (TankNumber==1) {
        //      if (Nobs>=1) Ncount++;
        // } else {
        //    Ncount += TankNumber * (1.-pow(1.*(TankNumber-1)/TankNumber,Nobs)); // exp value of number of tanks seeing a signal
        // }
        // NNBB The above calculation ignores correlations. We have removed it as we do not actually need to 
        // simulate a hard cut N_signal>=Ntrigger in the code, as we make it differentiable by using PActive[] instead.
        // ------------------------------------------------------------------------------------------------------------

        // Handle timing generation now
        // ----------------------------
        // All our signal stays within an integration window. We define the time response of a tank by taking the
        // weighted average of times for all observed muons and electrons separately. This means we need to define,
        // e.g. for muons, the average time of fbm*Nm from a Uniform, and the average time of (1-fbm)*Nm from a 
        // Gaussian, and do their weighted average.
        // ------------------------------------------------------------------------------------------------------
        // The flux model predicts mb muons and eb electrons from backgrounds, and m0, e0
        // from signal sources. These have different expected time distributions. Backgrounds should be sampled
        // from uniform, signal from gaussians. In a given event we observe a realization of nms, nes, nmb, neb and a 
        // corresponding time of these particles. Since we cannot simulate the time of each particle, we compute
        // a realization of the time we measure with them, using the average of the observed particles.
        // So we need to determine separately the average time of background and signal components, given the different
        // distributions the particles are sampled from, and their rms. We then sample from those assuming a Gaussian
        // distribution of each. We may then get the observed mean as a simple arithmetic mean given
        // the number of entries of the two components. This will be the "observed time" of the particles of each
        // kind. 
        // ------------------------------------------------------------------------------------------------------------
        double et = EffectiveTime(x[id],y[id],TrueX0[is],TrueY0[is],TrueTheta[is],TruePhi[is],0); // expected arrival time of signal
        // We assume that we can center the integration window around the expected arrival time of the signal
        // --------------------------------------------------------------------------------------------------
        if (Nm[id][is]>0) {
            std::pair<double,double> TAm = TimeAverage_SigPlusBgr(et,nms,nmb,true); // actual time observed for nms+nmb muons
            Tm[id][is]  = myRNG->Gaus(TAm.first,TAm.second);
            //Stm[id][is] = TAm.second; // expected time resolution of average time Tm given number of muons and the bgr fraction
        }
        if (Ne[id][is]>0) {
            std::pair<double,double> TAe = TimeAverage_SigPlusBgr(et,nes,neb,true); // actual time observed for nes+neb electrons
            Te[id][is] = myRNG->Gaus(TAe.first,TAe.second);
            //Ste[id][is] = TAe.second; // expected time resolution of average time Te given number of electrons and bgr fraction
        }
    } // end Nunit loop

    // Commented from v135 on, see note above. We use Active[] only for defining
    // showers that are reconstructable (i.e. likelihood converges, 2nd derivative pos-def)
    // ------------------------------------------------------------------------------------
    // if (Ncount>=Ntrigger) {
    //     Active[is]  = true;
    // }

    PActive[is] = ProbTrigger (SumProbGe1[is],0); // ProbTrigger (Ncounts,0);
    if (PActive[is]>0.) Active[is] = true;

    //if (is%100==0) cout << "is = " << is << " Ncount, SumProbGe1, pactive = " << Ncount << " " << SumProbGe1[is] << " " << PActive[is] << endl;
#ifdef PLOTS
    double minD = largenumber;
    for (int id=0;id<Nunits;id++) {
        double d = pow(x[id]-TrueX0[is],2.)+pow(y[id]-TrueY0[is],2.);
        if (d>0.) d = sqrt(d);
        if (d<minD) minD = d;
    }
    if (IsGamma[is]) {
        PAGvsD->Fill(minD,TrueE[is],PActive[is]);
        NPAGvsD->Fill(minD,TrueE[is],1.);
    } else {
        PAPvsD->Fill(minD,TrueE[is],PActive[is]);
        NPAPvsD->Fill(minD,TrueE[is],1.);
    }
#endif
    return;
}

// Set XY of showers in case they are not randomly resampled
// ---------------------------------------------------------
void SetShowersXY() {
    if (setXYto00) {
        for (int is=0; is<Nevents+Nbatch; is++) {
            TrueX0[is] = 0.;
            TrueY0[is] = 0.;
        }
    } else if (HexaShowers) {
        double r0 = sqrt(pow(TotalRspan,2)*pi/Nevents);
        TrueX0[0] = Xoffset;
        TrueY0[0] = Yoffset;
        int is   = 1;
        int n    = 6;
        double r = r0;
        do {
            for (int ith=0; ith<n && is<Nevents; ith++) {
                double phi = ith*twopi/n;
                TrueX0[is] = Xoffset + r*cos(phi);
                TrueY0[is] = Yoffset + r*sin(phi);
                //if (debug) cout << is << " ith=" << ith << " r=" << r << " x,y= " << TrueX0[is] << " " << TrueY0[is] << endl;
                is++;
            }
            n += 6;
            r += r0;
        } while (is<Nevents);

        if (SameShowers) {
            for (int is=Nevents; is<Nevents+Nbatch; is++) {
                TrueX0[is] = TrueX0[is-Nevents];
                TrueY0[is] = TrueY0[is-Nevents];
            }   
            Ntrials[is] = 1;
        } else {
            // And now do the same for the Nbatch events
            // -----------------------------------------
            TrueX0[Nevents] = Xoffset;
            TrueY0[Nevents] = Yoffset;
            is = Nevents+1;
            n = 6;
            r = r0;
            do {
                for (int ith=0; ith<n && is<Nevents+Nbatch; ith++) {
                    double phi = ith*twopi/n;
                    TrueX0[is] = Xoffset + r*cos(phi);
                    TrueY0[is] = Yoffset + r*sin(phi);
                    //if (debug) cout << is << " ith=" << ith << " r=" << r << " x,y= " << TrueX0[is] << " " << TrueY0[is] << endl;
                    is++;
                }
                n += 6;
                r += r0;
            } while (is<Nevents+Nbatch);
        }
    } else {
        // Below for a square grid of showers
        // ----------------------------------
        int side = sqrt(Nevents);
        for (int is=0; is<Nevents; is++) {
            TrueX0[is] = Xoffset -TotalRspan + 2.*TotalRspan*(is%side+0.5)/side;
            TrueY0[is] = Yoffset -TotalRspan + 2.*TotalRspan*(is/side+0.5)/side;
            //if (debug) cout << is << " x,y= " << TrueX0[is] << " " << TrueY0[is] << endl;
        }

        // Same, for Nbatch events 
        // -----------------------
        side = sqrt(Nbatch);
        for (int is=Nevents; is<Nevents+Nbatch; is++) {
            TrueX0[is] = Xoffset -TotalRspan + 2.*TotalRspan*((is-Nevents)%side+0.5)/side;
            TrueY0[is] = Yoffset -TotalRspan + 2.*TotalRspan*((is-Nevents)/side+0.5)/side;
            //if (debug) cout << is << " x,y= " << TrueX0[is] << " " << TrueY0[is] << endl;
        }
    }
}

// Generate shower and distribute particle signals in units
// --------------------------------------------------------
void GenerateShower (int is) {
 
    // Initialize counters etc.
    // ------------------------
    // int Ncount     = 0;
    // SumProbGe1[is] = 0.;
    // Active[is]     = true;
    // PActive[is]    = 0.;
    // Don't bother initializing them here - done in GetCounts()

    // For debugging purposes we may generate all the times the same shower
    // --------------------------------------------------------------------
    if (Debug_Rec) {
        TrueX0[is]    = 0.;
        TrueY0[is]    = 0.;
        Ntrials[is]   = 1;
        TrueE[is]     = 1.;
        TrueTheta[is] = 0.5;
        TruePhi[is]   = 0.5;
        GetCounts(is);
        return;
    }

    // Position of center of shower
    // Please remember that the generated distribution of showers must cover instrumented area
    // for all scanned configurations! We give some slack to the generated showers, such
    // that the system does not discover that the illuminated area has a step function
    // ---------------------------------------------------------------------------------------
    if (!fixShowerPos) { // Otherwise x0, y0 are defined in setshowersxy routine
        if (SameShowers && is>=Nevents) {
            TrueX0[is]  = TrueX0[is-Nevents]; // Note, this operation requires multithreading to be split for Nevents and Nbatch
            TrueY0[is]  = TrueY0[is-Nevents];
            Ntrials[is] = 1;
        } else {
            Ntrials[is] = 0; // We could get away with a single counter but for multithreading we need a vector

            // We want to simulate showers only closer to Rslack from each detector, and as detectors
            // get far away from one another this is tricky. But for the utility calculation, we only
            // need the density per square km2 generated, so we can keep track of this by accounting for
            // the number of trials we make to generate each shower (or all showers, but multithreading...)
            // --------------------------------------------------------------------------------------------
            double tryx0, tryy0;
            int iter = 0;
            bool out = true; 
            do {
                if (CircleExposure) {
                    // Illuminate circle on the ground 
                    // -------------------------------
                    double DfromCenter = sqrt(myRNG->Uniform(0., pow(TotalRspan,2)));
                    double phi = myRNG->Uniform(0.,twopi);
                    tryx0 = Xoffset + DfromCenter*cos(phi);  
                    tryy0 = Yoffset + DfromCenter*sin(phi);
                } else {
                    // Illuminate square area on the ground
                    // ------------------------------------
                    cout << "     Warning derivatives not properly computed for square exposure" << endl;
                    tryx0 = Xoffset -(TotalRspan)+myRNG->Uniform(2.*(TotalRspan));
                    tryy0 = Yoffset -(TotalRspan)+myRNG->Uniform(2.*(TotalRspan));
                }
                double r2;
                for (int id=0; id<Nunits && out; id++) {
                    r2 = pow(x[id]-tryx0,2)+pow(y[id]-tryy0,2);
                    if (r2<Rslack2) out = false; // Found a detector close enough, we take this shower
                }
                // Store trials for dudx,dy calculation, up to maxiter of them
                // -----------------------------------------------------------
                iter = Ntrials[is];
                if (iter<maxIter && out) {                         
                    TryX0[is][iter] = tryx0;
                    TryY0[is][iter] = tryy0;
                }
                Ntrials[is]++; // We count all trials even if we only store maxiter. The successful one is the Ntrials-th.
            } while (out); // enforce that this shower is not too far from at least a detector
            // Ok, we found a point within reach
            // ---------------------------------
            TrueX0[is] = tryx0;
            TrueY0[is] = tryy0;
        } // end if !sameshowers
    }  // end if !fixshowerpos

    // We get the number density per m^2 of muons and other particles
    // as a function of R at the nominal detector position, for all detector units.
    // Note that the density is computed _at_ the detector center: for wide detectors this becomes an approximation.
    // Also note, this matrix does not depend on energy - it is regenerated
    // for every energy point (we only use it inside the ie loop in the code calling this function)
    // -------------------------------------------------------------------------------------------------------------

    // We want a PDF f(E) = A*E^-B. To normalize it in [Emin,Emax] we need A/B = 1./(Emin^-B-Emax^-B).
    // We get the integral of the PDF as F(E) = A/B (Emin^-B-E^-B) = (Emin^-B-E^-B)/(Emin^-B-Emax^-B).
    // Thus if we generate rnd as uniform, we get E = pow(Emin^-B + rnd * (Emax^-B-Emin^-B),(-1/B)).
    // -----------------------------------------------------------------------------------------------
    if (fixE) {
        TrueE[is] = Efix; // Fixed energy
    } else {
        if (SameShowers) {
            // With SameShowers on, energy populates uniformly the spectrum, and theta, phi are set to zero
            // --------------------------------------------------------------------------------------------
            if (is<Nevents) {
                    TrueE[is] = Emin+(Emax-Emin)*(is+0.5)/Nevents;
            } else {
                    TrueE[is] = Emin+(Emax-Emin)*(is-Nevents+0.5)/Nbatch;
            }
        } else {
            if (Eslope!=0.) {
                TrueE[is] = pow(pow(Emin,-Eslope)+myRNG->Uniform()*(pow(Emax,-Eslope)-pow(Emin,-Eslope)),-1./Eslope);
            } else {
                TrueE[is] = Emin+(Emax-Emin)*myRNG->Uniform();
            }
        }
    }

    // Define polar and azimuthal angle of shower
    // ------------------------------------------
    if (OrthoShowers) {
        TrueTheta[is] = 0.;
        TruePhi[is]   = 0.;    
    } else if (SlantedShowers) { // Used to scan U_PS utility
        TruePhi[is]   = myRNG->Uniform(-pi,pi);
        TrueTheta[is] = pi/4.;
    } else {
        TruePhi[is]   = myRNG->Uniform(-pi,pi);
        // We used to generate a polar angle with a distribution cos(2theta), using
        // theta = 1/2 arcos(1.-1.6428r) with r in [0,1] uniform.
        // This came from taking the pdf P(theta) = sin(2 theta) which
        // comes from considering a uniform distribution in solid angle
        // of the sources, which gives a sin(theta) term, and an area term on the
        // ground, which gives a cos(theta) term. The cumulative of this distribution
        // is (1-cos(2theta))/2, which when normalized in [0,65 deg] gives f(theta) = 1-cos(2theta)/1.6428.
        // We inverted this to get r = 1/cos(2theta)/1.6428 -> theta = 1/2 arcos(1.-1.6428r)
        // 
        // Later we moved the ground cos(theta) term to the flux, and remained with a sin(theta) distribution.
        // For PeV source case, we have instead a uniform theta distribution in the source polar angle.
        // ---------------------------------------------------------------------------------------------------
        //do {
        //    TrueTheta[is] = 0.5*acos(1.-1.6428*myRNG->Uniform());
        //} while (TrueTheta[is]>thetamax-epsilon || TrueTheta[is]<=0.);
        if (!PeVSource || (PeVSource && !CrossingZenith)) {
            do {
                TrueTheta[is] = asin (myRNG->Uniform(0.,0.906308)); // sin65 deg
            } while (TrueTheta[is]>=thetamax-epsilon);  // do not bother to simulate too tilted showers
        } else {
            TrueTheta[is] = myRNG->Uniform(0.,thetamax); // for zenith-crosing PeV sources we generate a flat distribution
        }
        // TT->Fill(TrueTheta[is]);
    } // end if not SameShowers

    // Now get actual readouts in the detectors 
    // ----------------------------------------
    GetCounts (is); // This also fills SumProbGe1[]
    // cout << "For this shower: x0,y0 = " << TrueX0[is] << "," << TrueY0[is] << " th, phi = " << TrueTheta[is] << "," << TruePhi[is] << " E=" << TrueE[is] << endl;
    return;
}

// Poisson likelihood for position and angle of shower, given Nmu, Ne counts in Nunit detectors. Note,
// this calculation omits constant terms in the likelihood, because the latter is used in comparisons at fixed N or in ratios
// --------------------------------------------------------------------------------------------------------------------------
double ComputeLikelihood (int is, double x0, double y0, double theta, double phi, double energy, bool isgamma) {
    double logL = 0.;
    double ct   = cos(theta);
    for (int id=0; id<Nunits; id++) {
        int nm = Nm[id][is];
        int ne = Ne[id][is];
        double thisR = EffectiveDistance (x[id],y[id],x0,y0,theta,phi,0);
        double lambdaM, lambdaE;
        double fbm, fbe;
        if (isgamma) { // gamma HYPOTHESIS
            double m0 = MFromG (energy, theta, thisR, 0);
            double e0 = EFromG (energy, theta, thisR, 0);
            lambdaM = m0 * ct + fluxB_mu;
            lambdaE = e0 * ct + fluxB_e;
            fbm = fluxB_mu/lambdaM;
            fbe = fluxB_e/lambdaE;
        } else {        // proton HYPOTHESIS        
            double m0 = MFromP (energy, theta, thisR, 0);
            double e0 = EFromP (energy, theta, thisR, 0);
            lambdaM = m0 * ct + fluxB_mu;
            lambdaE = e0 * ct + fluxB_e;
            fbm = fluxB_mu/lambdaM;
            fbe = fluxB_e/lambdaE;
        }
        if (lambdaM<=0.) { // Zero total flux predictions - can happen if assumed X0,Y0 moved too far away
            if (nm>0.) return -largenumber;
        } else {
            logL -= lambdaM;
            logL += nm * log(lambdaM);
        }
        if (lambdaE<=0.) {
            if (ne>0.) return -largenumber;
        } else {
            logL -= lambdaE;
            logL += ne * log(lambdaE);
        }
        // We do not add the constant terms, see note below
        // ------------------------------------------------
        // logL += -LogFactorial(nm) - LogFactorial(ne);
        // The time variance results from a weighted sum, see routine AverageTime. However, we do not implement it yet below,
        // as the derivative of the likelihood terms is complex to calculate. To be done. 
        // ------------------------------------------------------------------------------------------------------------------
        double thisT = EffectiveTime (x[id],y[id],x0,y0,theta,phi,0); // the expected time is the same for electrons and muons
        if (nm>0) {
            // double eff_sigma_m = sqrt(((1.-fbm)*sigma_time*sigma_time + fbm*IntegrationWindow*IntegrationWindow/12)/Nm[id][is]);
            // double thisTm = EffectiveTime (x[id],y[id],x0,y0,theta,phi,0); // for now we do not implement the bgr effect
            //logL += -0.5*pow((Tm[id][is]-thisTm)/eff_sigma_m,2.); // - log(sqrt2pi*eff_sigma_m);
            logL += -0.5*pow((Tm[id][is]-thisT)/sigma_time,2.); // - log(sqrt2pi*sigma_time);
        }
        if (ne>0) {
            // double eff_sigma_e = sqrt(((1.-fbe)*sigma_time*sigma_time + fbe*IntegrationWindow*IntegrationWindow/12)/Ne[id][is]);
            // double thisTe = EffectiveTime (x[id],y[id],x0,y0,theta,phi,0);
            logL += -0.5*pow((Te[id][is]-thisT)/sigma_time,2.); // - log(sqrt2pi*sigma_time);
            // logL += -0.5*pow((Te[id][is]-thisTe)/eff_sigma_e,2.); // - log(sqrt2pi*eff_sigma_e);
        }
        // We omit the factorial term as we only use this likelihood in a ratio throughout, or only in comparison with fixed N.
    } // end id loop on Nunits
    return logL;
}

// Compute inverse variance of energy measurement for reconstructed photons
// ------------------------------------------------------------------------
double inv_rms_E (int is, int idet=0, int mode=0) {
    double d2lnLde2    = 0.;
    double dinvvarE_dr = 0.;
    double invrms;
    double em = e_meas[is][0];
    double tm = thmeas[is][0];
    float  ct = cos(tm);
    if (mode==0) {
        for (int id=0; id<Nunits; id++) {
            int nm = Nm[id][is];
            int ne = Ne[id][is];
            double thisR = EffectiveDistance (x[id],y[id],x0meas[is][0],y0meas[is][0],tm,phmeas[is][0],0);
            double lambdam = MFromG (em,tm,thisR,0)*ct + fluxB_mu;  
            double lambdae = EFromG (em,tm,thisR,0)*ct + fluxB_e;
            double dlmde   = MFromG (em,tm,thisR,2)*ct;
            double dlede   = EFromG (em,tm,thisR,2)*ct;
            double d2lmde2 = MFromG (em,tm,thisR,22)*ct;
            double d2lede2 = EFromG (em,tm,thisR,22)*ct;
            d2lnLde2 += -nm*pow(dlmde/lambdam,2.) +(nm/lambdam-1.)*d2lmde2;
            d2lnLde2 += -ne*pow(dlede/lambdae,2.) +(ne/lambdae-1.)*d2lede2;
            if (d2lnLde2!=d2lnLde2) cout << "is, id = " << is << " " << id << " d2lnlde2 += " 
                << -nm*pow(dlmde/lambdam,2.) +(nm/lambdam-1.)*d2lmde2 << " , "
                << -ne*pow(dlede/lambdae,2.) +(ne/lambdae-1.)*d2lede2 << " lambdam, lambdae = " << lambdam << " , " << lambdae << "dlde m,e = " << dlmde << " " << dlede << " d2le,m = " << d2lmde2 << " " << d2lede2 << endl;
        }
        // cout << " Total is d2lnLde2 = " << d2lnLde2 << " de/e = " << 100*(TrueE[is] - e_meas[is][0])/TrueE[is] << endl;
        if (d2lnLde2<0.) {
            invrms = sqrt(-d2lnLde2);
        } else {
            N_pos_derivative++;
            invrms = maxinvrms; // Signals not well reconstructed 
            Active[is]  = false; 
            PActive[is] = 0.;
            // cout << "     negative -d2lnLde2" << endl;
        }
        return invrms;
    } else if (mode==1) { // Return derivative wrt R for detector idet
        int nm = Nm[idet][is];
        int ne = Ne[idet][is];
        double thisR     = EffectiveDistance(x[idet],y[idet],x0meas[is][0],y0meas[is][0],tm,phmeas[is][0],0);
        double lambdam   = MFromG (em,tm,thisR,0)*ct + fluxB_mu;
        double lambdae   = EFromG (em,tm,thisR,0)*ct + fluxB_e;
        double dlmdr     = MFromG (em,tm,thisR,1)*ct;
        double dledr     = EFromG (em,tm,thisR,1)*ct;
        double dlmde     = MFromG (em,tm,thisR,2)*ct;
        double dlede     = EFromG (em,tm,thisR,2)*ct;
        double d2lmde2   = MFromG (em,tm,thisR,22)*ct;
        double d2lede2   = EFromG (em,tm,thisR,22)*ct;
        double d2lmdedr  = MFromG (em,tm,thisR,23)*ct;
        double d2lededr  = EFromG (em,tm,thisR,23)*ct;
        double d3lmd2edr = MFromG (em,tm,thisR,24)*ct;
        double d3led2edr = EFromG (em,tm,thisR,24)*ct;
        double d3lmde3   = MFromG (em,tm,thisR,25)*ct;
        double d3lede3   = EFromG (em,tm,thisR,25)*ct;
        double dedr      = dEk_dRik(idet,is);
        //cout << " For id = " << idet << " R= " << thisR << " Dedr is " << dedr << endl;
        double lm2       = pow(lambdam,2.);
        double lm3       = lm2*lambdam;
        double dlmdesq   = pow(dlmde,2.);
        double le2       = pow(lambdae,2.);
        double le3       = le2*lambdae;
        double dledesq   = pow(dlede,2.);        
        /*dinvvarE_dr += 2.*nm/lm3*dlmdr*dlmdesq 
                       - nm/lm2 * (2.*dlmde*d2lmdedr + dlmdr*dlmdesq) 
                       + (nm/lambdam-ct)*d3lmdrd2e -nm/lm2*dlmdr*d2lmde2;
        dinvvarE_dr += 2.*ne/le3*dledr*dledesq 
                       - ne/le2 * (2.*dlede*d2lededr + dledr*dledesq) 
                       + (ne/lambdae-ct)*d3ledrd2e -ne/le2*dledr*d2lede2;;
        return -dinvvarE_dr/(2.*invrms); // above we derived over dr the d2lnLde2, but we need a negative sign to use it as an estimate of 1/sigma2
        */

        // Mathematica solution, May 2 - d(inv_rms_E)/dE(R)*dE(R)/dR + d(inv_rms_E)/dR.
        // ----------------------------------------------------------------------------
        double mathsol_full = - 2.*ne*dledesq*(dledr+dedr*dlede)/le3 
                              - 2.*nm*dlmdesq*(dlmdr+dedr*dlmde)/lm3  
                              + ne*(dledr*d2lede2+dlede*(2.*d2lededr+3.*dedr*d2lede2))
                              + nm*(dlmdr*d2lmde2+dlmde*(2.*d2lmdedr+3.*dedr*d2lmde2))
                              - ne*(d3led2edr+dedr*d3lede3) - nm*(d3lmd2edr+dedr*d3lmde3)
                              + ct*(d3led2edr+d3lmd2edr+dedr*(d3lede3+d3lmde3));
        //if (fabs(mathsol_full)>largenumber) cout << " idet= " << idet << " R = " << thisR<< " dinvsigede=" << mathsol_full << " ne = " << ne 
        //                                      << " nm = " << nm << " dledesq = " << dledesq << " dlmdesq = " << dlmdesq << " dledr = " << dledr << " dlmdr = " << dlmdr << " le3= " 
        //                                      << le3 << " lm3 = " << lm3 << " dedr = " << dedr << " d3led2edr= " << d3led2edr << " d3lmd2edr" << d3lmd2edr 
        //                                      << " d3lede = " << d3lede3 << " d3lmde = " << d3lmde3 << " dlede = " << dlede << endl;
        return mathsol_full;
    } 
    return epsilon;
}

// Find most likely position of shower center by max likelihood
// ------------------------------------------------------------
double FitShowerParams (int is, bool GammaHyp) {

    bool FalseHypothesis = true;
    if ((IsGamma[is] && GammaHyp) || (!IsGamma[is] && !GammaHyp)) FalseHypothesis = false;

    double currentX0;
    double currentY0;
    double currentTheta;
    double currentPhi;
    double currentE;
    double logL_in_start_true = - largenumber;

    // Bypass all initialization if we use true values for it
    // ------------------------------------------------------
    double maxlogL_in = -largenumber;
    if (initTrueVals || CheckInitialization) {
        currentX0    = TrueX0[is];
        currentY0    = TrueY0[is];
        currentTheta = TrueTheta[is];
        currentPhi   = TruePhi[is];
        currentE     = TrueE[is];
        if (CheckInitialization) {
            logL_in_start_true = ComputeLikelihood (is, currentX0, currentY0, currentTheta, currentPhi, currentE, GammaHyp);
        }
    } 
    // We do a grid search for the initialization point of the likelihood:
    // - always, if this is a gamma and we fit for a proton or vice-versa;
    // - in all cases, when initTrueVals is false.
    // -------------------------------------------------------------------
    if (!initTrueVals || FalseHypothesis) {
        // Initialize shower position at max flux
        // --------------------------------------
        currentX0    = 0.;
        currentY0    = 0.;
        currentTheta = 0.5;
        currentPhi   = 0.;
        if (PeVSource) {
            currentE = E_PS;
        } else {
            currentE = 1.; // Just a first guess, midpoint of log_10 E between 100 TeV and 10 PeV - NB to be updated if we change the range
        }

        // Preliminary assay of grid of points
        // -----------------------------------
        double cX0, cY0, cT0, cP0, cE0;
        cX0 = currentX0;
        cY0 = currentY0;
        cT0 = currentTheta;
        cP0 = currentPhi;
        cE0 = currentE;
        int N_g = sqrt(1.*Ngrid);
        int N_p = 4;
        int N_t = 4;  
        int N_e = NEgrid;
        if (usetrueXY) {
            N_g  = 1;
            currentX0 = TrueX0[is];
            currentY0 = TrueY0[is];
            cX0 = currentX0;
            cY0 = currentY0;
        }
        if (usetrueAngs) {
            N_p  = 1; 
            N_t  = 1;
            currentTheta = TrueTheta[is];
            currentPhi   = TruePhi[is];
            cT0 = currentTheta;
            cP0 = currentPhi;
        } 
        if (usetrueE) {
            N_e = 1;
            currentE = TrueE[is];
            cE0 = currentE;
        }

        // We might want to fit for parameters but force initialization to true values. We do it below
        // -------------------------------------------------------------------------------------------
        int currBitmap = initBitmap;
        if (currBitmap>=16) {
            currentX0 = TrueX0[is];
            currBitmap -= 16;
            cX0 = currentX0;
        }
        if (currBitmap>=8) {
            currentY0 = TrueY0[is];
            currBitmap -= 8;
            cY0 = currentY0;
        }
        if (currBitmap>=4) {
            currentTheta = TrueTheta[is];
            currBitmap -= 4;
            cT0 = currentTheta;
        } 
        if (currBitmap>=2) {
            currentPhi = TruePhi[is];
            currBitmap -= 2;
            cP0 = currentPhi;
        }
        if (currBitmap==1) {
            currentE = TrueE[is];
            cE0 = currentE; // We need this in loop for xye below
        }
        //cout << is << " " << currBitmap << " " << cE0 << " " << TrueE[is] << endl;
        if (initBitmap%8==1 && !usetrueAngs) { // then we do not initialize theta, phi values to true ones.
            // We find suitable angles first, as they are identifiable even without precise E, X0, Y0
            // --------------------------------------------------------------------------------------
            for (int ip=0; ip<N_p; ip++) {
                if (N_p>1) cP0 = (-1.+(1.+2.*ip)/N_p)*pi;
                for (int it=0; it<N_t; it++) {
                    if (N_t>1) cT0 = ((0.5+it)/N_t)*(halfpi-pi/8.);
                    // Compute likelihood for this point
                    double logL_in = ComputeLikelihood (is, currentX0, currentY0, cT0, cP0, currentE, GammaHyp);
                    if (logL_in>maxlogL_in) {
                        currentTheta = cT0;
                        currentPhi   = cP0;
                        maxlogL_in   = logL_in;
                    }
                }                
            }
        }
        
        // Now loop to find best point in X0,Y0,E
        // --------------------------------------
        if (initBitmap<8 && !usetrueXY) { // Then we do not initialize x0,y0 to true values.
            maxlogL_in = -largenumber;
            for (int ie=0; ie<N_e; ie++) {
                if (N_e>1) cE0 = Einit[ie]; // If initBitmap%2=1 we have already initialized it 
                for (int ix=0; ix<N_g; ix++) {
                    if (N_g>1) cX0 = (-1.+(1.+2.*ix)/N_g)*(TotalRspan);
                    for (int iy=0; iy<N_g; iy++) {
                        if (N_g>1) cY0 = (-1.+(1.+2.*iy)/N_g)*(TotalRspan);
                        // Compute likelihood for this point
                        double logL_in = ComputeLikelihood (is, cX0, cY0, currentTheta, currentPhi, cE0, GammaHyp);
                        //if (GammaHyp==IsGamma[is] && is%100<2) cout << "ie,ix,iy = " << ie << " " << ix << " " << iy 
                        //                                         << " e,x,y =" << cE0 << " " << cX0 << " " << cY0 
                        //                                          << " et,xt,yt = " << TrueE[is] << 
                        //                                          " " << TrueX0[is] << " " << TrueY0[is] << " logL = " << logL_in << endl;
                        if (logL_in>maxlogL_in) {
                            currentX0    = cX0;
                            currentY0    = cY0;
                            currentE     = cE0;
                            maxlogL_in   = logL_in;
                        }
                    }
                }
            }
        }
    } // End if use true vals for init
    if (CheckInitialization) {
        Start_true_trials++;
        if (maxlogL_in<=logL_in_start_true) Start_true_wins++; 
    }

    // Declare and define variables used in loop below
    // -----------------------------------------------
    double dlogLdX0;
    double dlogLdY0;
    double dlogLdTh;
    double dlogLdPh;
    double dlogLdE;
    double logL           = 0.;
    double prevlogL       = 0.;
    double prev2logL      = 0.;
    double LearningRateX  = LRX;  
    double LearningRateY  = LRX; 
    double LearningRateTh = LRA;
    double LearningRatePh = LRA; 
    double LearningRateE  = LRE; 
    int istep  = 0;
    double m_x = 0.;
    double v_x = 0.;
    double m_y = 0.;
    double v_y = 0.;
    double m_t = 0.;
    double v_t = 0.;
    double m_p = 0.;
    double v_p = 0.;
    double m_e = 0.;
    double v_e = 0.;
    float lastXincr = 0.;
    float lastYincr = 0.;
    float lastEincr = 0.;
    float lastTincr = 0.;
    float lastPincr = 0.;

    // Compute sum of constant terms in logL
    // -------------------------------------
    double logLfix = 0.;
    for (int id=0; id<Nunits; id++) { 
        logLfix -= LogFactorial (Nm[id][is]);
        logLfix -= LogFactorial (Ne[id][is]);
    }

    // Loop to maximize logL and find X0, Y0, Theta, Phi, E of shower
    // --------------------------------------------------------------
    do {
        prev2logL  = prevlogL;
        prevlogL   = logL;
        logL       = logLfix; // We include this in the calculation, so that we can appraise the reconstruction performance 
        dlogLdX0   = 0.;
        dlogLdY0   = 0.;
        dlogLdTh   = 0.;
        dlogLdPh   = 0.;
        dlogLdE    = 0.;
        double ct  = cos(currentTheta);

        // Sum contributions from all detectors to logL and derivatives
        // ------------------------------------------------------------
        for (int id=0; id<Nunits; id++) {
            int nm   = Nm[id][is];
            int ne   = Ne[id][is];
            float tm = Tm[id][is];
            float te = Te[id][is];
            // How far is this unit from assumed shower center, projected along direction?
            // ---------------------------------------------------------------------------
            double thisR = EffectiveDistance (x[id], y[id], currentX0, currentY0, currentTheta, currentPhi, 0);
            double tmpt  = EffectiveTime     (x[id], y[id], currentX0, currentY0, currentTheta, currentPhi, 0);
            double lambdaM0, lambdaE0;
            if (GammaHyp) { // gamma HYPOTHESIS
                lambdaM0 = MFromG (currentE, currentTheta, thisR, 0);
                lambdaE0 = EFromG (currentE, currentTheta, thisR, 0);
            } else {        // proton HYPOTHESIS        
                lambdaM0 = MFromP (currentE, currentTheta, thisR, 0);
                lambdaE0 = EFromP (currentE, currentTheta, thisR, 0);
            }
            double lambdaM = lambdaM0 * ct + fluxB_mu; // For tilted showers the flux is reduced, as the cross section of the tank is
            double lambdaE = lambdaE0 * ct + fluxB_e;  // (here we are not modeling the full cylinder, which would modify this simple picture)
            double fbm = fluxB_mu/lambdaM;
            double fbe = fluxB_e/lambdaE;
            // In this version we do not implement the background dependence on the time 
            // -------------------------------------------------------------------------
            //double thisTm = tmpt; // (1.-fbm)*tmpt; // the bgr piece averages to zero
            //double thisTe = tmpt; // (1.-fbe)*tmpt; // the bgr piece averages to zero
            if (lambdaM<=0.) { // zero total flux predictions 
                if (nm>0.) logL -= largenumber;
            } else {
                logL += -lambdaM + nm*log(lambdaM);
            }
            if (lambdaE<=0.) {
                 if (ne>0.) logL -= largenumber;
            } else {
                logL += -lambdaE + ne*log(lambdaE);
            }
            // Add Gaussian term depending on detected arrival time t of particles in detector.
            // The sigma is the expected time resolution given the observed number of particles. It comes from combining 
            // the flat pdf of the background with the gaussian for the signal part, as done in routine AverageTime().
            // Note: this term only exists if the number of particles is >0, but the Poisson terms still exist in that case.
            // ------------------------------------------------------------------------------------------------------------------
            if (nm>0.) {
                // float eff_sigma_tm = Stm[id][is]; // sqrt(((1.-fbm)*sigma_time*sigma_time + fbm*IntegrationWindow*IntegrationWindow/12.)/nm);
                logL += -0.5*pow((tm-tmpt)/sigma_time,2.) - log(sqrt2pi*sigma_time);
                //logL += -0.5*pow((tm-tmpt)/eff_sigma_tm,2.) - log(sqrt2pi*eff_sigma_tm);
            }
            if (ne>0.) {
                // float eff_sigma_te = Ste[id][is]; // sqrt(((1.-fbe)*sigma_time*sigma_time + fbe*IntegrationWindow*IntegrationWindow/12.)/ne);
                logL += -0.5*pow((te-tmpt)/sigma_time,2.) - log(sqrt2pi*sigma_time);
                //logL += -0.5*pow((te-tmpt)/eff_sigma_te,2.) - log(sqrt2pi*eff_sigma_te);
            }
            if (logL!=logL) {
                cout << " Warning fsp: id = " << id << " E = " << currentE << " thisR = " << thisR << " logL = " << logL 
                     << " " << -lambdaM << " " << -lambdaE << " " << nm << " " << ne << " "
                     << -nm*log(lambdaM) << " " << -ne*log(lambdaE) << endl;
                cout << currentX0 << " " << currentY0 << " " << currentTheta << " " << currentPhi << " " << currentE << endl;
                cout << x[id] << " " << y[id] << " " << tm << " " << te << endl;
                outfile << " Warning fsp: id = " << id << " E = " << currentE << " thisR = " << thisR << " logL = " << logL 
                     << " " << -lambdaM << " " << -lambdaE << " " << nm << " " << ne << " "
                     << -nm*log(lambdaM) << " " << -ne*log(lambdaE) << endl;
                outfile << currentX0 << " " << currentY0 << " " << currentTheta << " " << currentPhi << " " << currentE << endl;
                outfile << x[id] << " " << y[id] << " " << tm << " " << te << endl;
                warnings3++;
                SaveLayout();
                return 0.;
            }

            double dlM0dR, dlE0dR, dlogLdR;
            if (!usetrueXY || !usetrueAngs) { // These calcs are used in both cases
                if (GammaHyp) { // gamma HYP
                    dlM0dR = MFromG (currentE, currentTheta, thisR, 1);
                    dlE0dR = EFromG (currentE, currentTheta, thisR, 1);
                } else {        // proton HY1P
                    dlM0dR = MFromP (currentE, currentTheta, thisR, 1); 
                    dlE0dR = EFromP (currentE, currentTheta, thisR, 1); 
                }
                // Compute derivative of logL with respect to R and E.
                // Since log L = -(lambda0*ct+TA) +N*log (lambda0*ct+TA) + cost,
                // dlogL/dR = -ct*dl0dR + N/lambda * dl0dR*ct .
                // The same calculation works for dlogL/dE
                // -------------------------------------------------------
                dlogLdR = - (dlM0dR + dlE0dR)*ct;
                if (lambdaM0>0.) {
                    dlogLdR += nm/lambdaM * ct*dlM0dR;  
                    // Add contribution from background term in p(time)
                    dlogLdR += -fbm*ct/lambdaM*dlM0dR;
                }
                if (lambdaE0>0.)  {
                    dlogLdR += ne/lambdaE * ct*dlE0dR;  
                    // Add contribution from background term in p(time)
                    dlogLdR += -fbe*ct/lambdaE*dlE0dR;
                }
            }
            // The two terms above are only needed if !usetrueXY or !usetrueangs
            // -----------------------------------------------------------------
            double factorm = 0.;
            double factore = 0.;
            // if (nm>0.) factorm = (1-fbm)*(tm-thisTm)/(c0*eff_sigma_m*eff_sigma_m*den);
            // if (ne>0.) factore = (1-fbe)*(te-thisTe)/(c0*eff_sigma_e*eff_sigma_e*den);
            // no bgr dep on time for now:
            if (nm>0.) factorm = (tm-tmpt)/(c0*sigma2_time);
            if (ne>0.) factore = (te-tmpt)/(c0*sigma2_time);
            if (!usetrueXY) {
                // Finally get dlogL/dx and dy from dlogLdR
                // ----------------------------------------
                double dRdX0 = EffectiveDistance (x[id], y[id], currentX0, currentY0, currentTheta, currentPhi, 1);
                double dRdY0 = EffectiveDistance (x[id], y[id], currentX0, currentY0, currentTheta, currentPhi, 2);
                dlogLdX0 += dRdX0 * dlogLdR;
                dlogLdY0 += dRdY0 * dlogLdR;
                // Add contribution from dlogL/dthisT * dthisT/dX0,Y0. We get it from
                // dlogL/dthisT = 0.5*2*(t-thisT)/sigma2t  and  dthisT/dX = -(sin(theta)cos(phi)/c0,
                // ---------------------------------------------------------------------------------
                dlogLdX0 += (factorm+factore)*(-sin(currentTheta)*cos(currentPhi));
                dlogLdY0 += (factorm+factore)*(-sin(currentTheta)*sin(currentPhi));
            }
            if (!usetrueE) {
                double dlM0dE, dlE0dE;
                if (GammaHyp) { // gamma HYP
                    dlM0dE = MFromG (currentE, currentTheta, thisR, 2);
                    dlE0dE = EFromG (currentE, currentTheta, thisR, 2);
                } else {        // proton HYP
                    dlM0dE = MFromP (currentE, currentTheta, thisR, 2);
                    dlE0dE = EFromP (currentE, currentTheta, thisR, 2);
                }
                // As above, the cos(theta) factor only plays in the -1 part from the derivative dlogL/dlambda
                // -------------------------------------------------------------------------------------------
                dlogLdE -= (dlM0dE + dlE0dE)*ct;
                if (lambdaM>0.) {
                    dlogLdE += nm/lambdaM * dlM0dE*ct; 
                }
                if (lambdaE>0.)  {
                    dlogLdE += ne/lambdaE * dlE0dE*ct;
                }
            }
            if (!usetrueAngs) {
                // Also get dlogL/dtheta and dlogL/dphi
                // NB to handle the indirect dependence of lambdas on theta through R, we compute
                // this part first, and then use it in the dflux/dtheta calculations below (mode=3)
                // NB it would have been nicer to declare it static, but then multithreading would
                // mess it up...
                // --------------------------------------------------------------------------------
                double dRdTh = EffectiveDistance (x[id], y[id], currentX0, currentY0, currentTheta, currentPhi, 3);
                double dRdPh = EffectiveDistance (x[id], y[id], currentX0, currentY0, currentTheta, currentPhi, 4);

                double st    = sin(currentTheta);
                double sp    = sin(currentPhi);
                double cp    = cos(currentPhi);
                double dx    = x[id]-currentX0;
                double dy    = y[id]-currentY0;

                // Since (NB updated after intro of background)
                //    log L = -(lambda0*ct+TA) + N * log (lambda0*ct+TA) + cost,
                //    dlogL/dth = -ct*dl0dth + st*lambda0 + N/(lambda0*ct+TA) * (dl0dth*ct - lambda0*st)
                // which becomes
                //    dlogL/dth = dl0dth*ct*(N/lambda-1) + st*(lambda0-lambda0*N/lambda) =
                //              = dl0dth*ct*(N/lambda-1) + st*lambda0*(1-N/lambda) = 
                //              = (N/lambda-1)*(dl0dth*ct-lambda0*st)
                // ----------------------------------------------------------------------------------
                double dlM0dth, dlE0dth;
                // Note that calculations below (mode=3) include use of dRdTh computed above.
                if (GammaHyp) { // gamma HYP
                    dlM0dth = MFromG (currentE, currentTheta, thisR, 3, dRdTh);
                    dlE0dth = EFromG (currentE, currentTheta, thisR, 3, dRdTh);
                } else {        // proton HYP
                    dlM0dth = MFromP (currentE, currentTheta, thisR, 3, dRdTh);
                    dlE0dth = EFromP (currentE, currentTheta, thisR, 3, dRdTh);
                }
                if (lambdaM>0.) dlogLdTh += (nm/lambdaM-1.) * (dlM0dth*ct - st*lambdaM0);
                if (lambdaE>0.) dlogLdTh += (ne/lambdaE-1.) * (dlE0dth*ct - st*lambdaE0); 

                dlogLdPh += dRdPh * dlogLdR;

                // Contributions from dL/dthisT * dthisT/dTh, dthisT/dPh:
                // Again, only present if nm, ne are positive
                // ------------------------------------------------------
                if (nm>0.) {
                    dlogLdTh += factorm *ct*(cp*dx+sp*dy);
                    dlogLdPh += factorm *st*(-sp*dx+cp*dy);
                }
                if (ne>0.) {
                    dlogLdTh += factore *ct*(cp*dx+sp*dy);
                    dlogLdPh += factore *st*(-sp*dx+cp*dy);
                }
            }
        } // end id loop on Nunits

        // Take a step in X0, Y0
        // ---------------------
        if (!usetrueXY) { 
            // ADAM GD rule:
            // -------------
            m_x = beta1 * m_x + (1.-beta1)*dlogLdX0;
            v_x = beta2 * v_x + (1.-beta2)*pow(dlogLdX0,2);
            double m_x_hat = m_x/(1.-powbeta1[istep+1]);
            double v_x_hat = v_x/(1.-powbeta2[istep+1]);
            double incr = LearningRateX * m_x_hat / (sqrt(v_x_hat)+epsilon); 
            if (fabs(incr)>epsilon) currentX0 += incr;
            lastXincr = incr;
            m_y = beta1 * m_y + (1.-beta1)*dlogLdY0;
            v_y = beta2 * v_y + (1.-beta2)*pow(dlogLdY0,2);
            double m_y_hat = m_y/(1.-powbeta1[istep+1]);
            double v_y_hat = v_y/(1.-powbeta2[istep+1]);
            incr = LearningRateY * m_y_hat / (sqrt(v_y_hat)+epsilon);
            if (fabs(incr)>epsilon) currentY0 += incr;
            lastYincr = incr;
        }

        // Also take a step in theta and phi
        // ---------------------------------
        if (!usetrueAngs) {
            // ADAM GD rule:
            // -------------
            m_t = beta1 * m_t + (1.-beta1)*dlogLdTh;
            v_t = beta2 * v_t + (1.-beta2)*pow(dlogLdTh,2);
            double m_t_hat = m_t/(1.-powbeta1[istep+1]);
            double v_t_hat = v_t/(1.-powbeta2[istep+1]);
            double incr = LearningRateTh * m_t_hat / (sqrt(v_t_hat)+epsilon);
            if (fabs(incr)>epsilon) currentTheta += incr;
            lastTincr = incr;
            m_p = beta1 * m_p + (1.-beta1)*dlogLdPh;
            v_p = beta2 * v_p + (1.-beta2)*pow(dlogLdPh,2);
            double m_p_hat = m_p/(1.-powbeta1[istep+1]);
            double v_p_hat = v_p/(1.-powbeta2[istep+1]);
            incr = LearningRatePh * m_p_hat / (sqrt(v_p_hat)+epsilon);
            if (fabs(incr)>epsilon)     currentPhi += incr;
            if (currentTheta>=thetamax) currentTheta = thetamax-epsilon; // Hard reset if hitting boundary
            if (currentTheta<=0.)       currentTheta = epsilon;          // Hard reset if hitting boundary
            lastPincr = incr;
        }

        // And a step in E
        // ---------------
        if (!usetrueE) {
            // ADAM GD rule:
            // -------------
            m_e = beta1 * m_e + (1.-beta1)*dlogLdE;
            v_e = beta2 * v_e + (1.-beta2)*pow(dlogLdE,2);
            double m_e_hat = m_e/(1.-powbeta1[istep+1]);
            double v_e_hat = v_e/(1.-powbeta2[istep+1]);
            double incr = LearningRateE * m_e_hat / (sqrt(v_e_hat)+epsilon); 
            if (fabs(incr)>epsilon) currentE += incr;
            if (currentE<=0.1) currentE = 0.1+epsilon; // Very loose requirement on reconstruction
            if (currentE>=10.) currentE = 10.-epsilon; // Can't exceed these boundaries or fluxes are not computable! (we should extend them analytically but for now that is that)
            lastEincr = incr;
        }
        istep++;

        /*if ( istep%10==0 && (IsGamma[is] && GammaHyp || (!IsGamma[is] && !GammaHyp))) {
            cout << "  istep = " << istep << " logL = " << logL << " xt,yt = " << TrueX0[is] << "," << TrueY0[is] << " xm,ym = " << currentX0 << "," << currentY0 
                 << " Et = " << TrueE[is] << " Em = " << currentE << " Tt,Pt = " << TrueTheta[is] << "," << TruePhi[is] << " Tm,Pm = " << currentTheta << "," << currentPhi << endl;
        }*/

    } while (istep<Nsteps && (istep<3 || fabs(logL-prevlogL)>logLApprox || fabs(prevlogL-prev2logL)>logLApprox)); // set it to 0.5 by default 

    if (IsGamma[is] && GammaHyp) {
        AverLastXIncr += lastXincr;
        AverLastYIncr += lastYincr;
        AverLastEIncr += lastEincr/currentE;
        AverLastTIncr += lastTincr;
        AverLastPIncr += lastPincr;
        NAverLastIncr++;
    }
    NumAvgSteps+=istep;
    DenAvgSteps++;

#ifdef FEWPLOTS
    //if (IsGamma[is] && GammaHyp || (!IsGamma[is] && !GammaHyp)) {
    //    NumStepsvsxy->Fill(sqrt(pow(TrueX0[is],2)+pow(TrueY0[is],2)),TrueE[is],(double)istep);
    //    NumStepsvsxyN->Fill(sqrt(pow(TrueX0[is],2)+pow(TrueY0[is],2)),TrueE[is]);
    //}
#endif
#ifdef PLOTS
    if (IsGamma[is] && GammaHyp) NumStepsg->Fill(TrueE[is],(double)istep);
    if (!IsGamma[is] && !GammaHyp) NumStepsp->Fill(TrueE[is],(double)istep); 
#endif
    //if ( (IsGamma[is] && GammaHyp || (!IsGamma[is] && !GammaHyp))) {
    //    cout << "  istep = " << istep << " logL = " << logL << " xt,yt = " << TrueX0[is] << "," << TrueY0[is] << " xm,ym = " << currentX0 << "," << currentY0 
    //            << " Et = " << TrueE[is] << " Em = " << currentE << " Tt,Pt = " << TrueTheta[is] << "," << TruePhi[is] << " Tm,Pm = " << currentTheta << "," << currentPhi << endl;
    //}

    if (currentPhi>pi) currentPhi  -= twopi;
    if (currentPhi<-pi) currentPhi += twopi;
    //if (debug) {
    //    if (IsGamma[is] && GammaHyp && Active[is]) {
    //        cout << " x,y = " << currentX0 << "," << currentY0 << " E = " << currentE << " true E = " << TrueE[is];
    //        if (SameShowers && is>=Nevents) cout << " Previous reco E = " << e_meas[is-Nevents][0];
    //        cout << endl;
    //    }
    //}

    //if (!usetrueAngs) cout << "true th = " << TrueTheta[is] << " meas = " << currentTheta << " true ph = " << TruePhi[is] << " meas = " << currentPhi << endl;
    //if (!usetrueXY)   cout << "true x = " << TrueX0[is] << " meas = " << currentX0 << " true y = << " << TrueY0[is] << " meas = " << currentY0 << endl;

    // Now we have the estimates of X0, Y0, and the logLR at max for event is 
    // ----------------------------------------------------------------------
#ifdef PLOTS
    double Rfrom00 = pow(TrueX0[is],2)+pow(TrueY0[is],2);
    Rfrom00 = sqrt(Rfrom00+epsilon);
    int iR = (int)(Rfrom00/1500.*NbinsResR);
    if (iR>NbinsResR-1) iR = NbinsResR-1;
    int iE = (int)(TrueE[is]/10.*NbinsResE);
    if (iE>NbinsResE-1) iE = NbinsResE-1;
    int ih = iR*NbinsResE+iE;
    if (GammaHyp && IsGamma[is]) { // gamma
        double delta = currentX0-TrueX0[is];
        if (fabs(delta)<maxdxy) {
            DXG->Fill(delta);
            DXYg[ih]->Fill(fabs(delta));
        } else {
            if (delta>0.) {
                DXG->Fill(maxdxy-epsilon);
                DXYg[ih]->Fill(fabs(maxdxy-epsilon));
            }
            if (delta<0.) {
                DXG->Fill(-maxdxy+epsilon);
                DXYg[ih]->Fill(fabs(-maxdxy+epsilon));
            }
        }
        delta = currentY0-TrueY0[is];
        if (fabs(delta)<maxdxy) {
            DYG->Fill(delta);
            DXYg[ih]->Fill(fabs(delta));
        } else {
            if (delta>0.) {
                DYG->Fill(maxdxy-epsilon);
                DXYg[ih]->Fill(fabs(maxdxy-epsilon));
            }
            if (delta<0.) {
                DYG->Fill(-maxdxy+epsilon);
                DXYg[ih]->Fill(fabs(-maxdxy+epsilon));
            }
        }
        DTHG->Fill(currentTheta-TrueTheta[is]);
        DT_g[ih]->Fill(fabs(currentTheta-TrueTheta[is]));
        DTHGvsT->Fill(currentTheta-TrueTheta[is],TrueTheta[is]);
        double dp = currentPhi - TruePhi[is];
        if (dp>2.*pi)  dp -= twopi;
        if (dp<-2.*pi) dp += twopi;
        DPHG->Fill(dp);
        DP_g[ih]->Fill(fabs(dp));
        DEG->Fill(fabs(currentE-TrueE[is])/TrueE[is]);
        DE_g[ih]->Fill(fabs(currentE-TrueE[is])/TrueE[is]);
    } else if (!GammaHyp && !IsGamma[is]) { // proton
        double delta = currentX0-TrueX0[is];
        if (fabs(delta)<maxdxy) {
            DXP->Fill(delta);
            DXYp[ih]->Fill(fabs(delta));
        } else { 
            if (delta>0.) {
                DXP->Fill(maxdxy-epsilon);
                DXYp[ih]->Fill(fabs(maxdxy-epsilon));
            }
            if (delta<0.) {
                DXP->Fill(-maxdxy+epsilon);
                DXYp[ih]->Fill(fabs(-maxdxy+epsilon));
            }
        }
        delta = currentY0-TrueY0[is];
        if (fabs(delta)<maxdxy) {
            DYP->Fill(delta);
            DXYp[ih]->Fill(fabs(delta));
        } else {
            if (delta>0.) {
                DYP->Fill(maxdxy-epsilon);
                DXYp[ih]->Fill(fabs(maxdxy-epsilon));
            }
            if (delta<0.) {
                DYP->Fill(-maxdxy+epsilon);
                DXYp[ih]->Fill(fabs(-maxdxy+epsilon));
            }
        }
        DTHP->Fill(currentTheta-TrueTheta[is]);
        DT_p[ih]->Fill(fabs(currentTheta-TrueTheta[is]));
        DTHPvsT->Fill(currentTheta-TrueTheta[is],TrueTheta[is]);
        double dp = currentPhi - TruePhi[is];
        if (dp>2.*pi)  dp -= twopi;
        if (dp<-2.*pi) dp += twopi;
        DPHP->Fill(dp);
        DP_p[ih]->Fill(fabs(dp));
        DEP->Fill(fabs(currentE-TrueE[is])/TrueE[is]);
        DE_p[ih]->Fill(fabs(currentE-TrueE[is])/TrueE[is]);
    } 
#endif
    if (GammaHyp) {
        x0meas[is][0] = currentX0;
        y0meas[is][0] = currentY0;
        thmeas[is][0] = currentTheta;
        phmeas[is][0] = currentPhi;
        e_meas[is][0] = currentE;
    } else {
        x0meas[is][1] = currentX0;
        y0meas[is][1] = currentY0;
        thmeas[is][1] = currentTheta;
        phmeas[is][1] = currentPhi;
        e_meas[is][1] = currentE;
    }
    if (logL!=logL) {
        cout << "Problems with logL, return -largenumber" << endl;
        logL = -largenumber;
        SaveLayout();
        return 0.;
    }

    // We compute the inverse rms of E measurement so that we know if we have to discard this shower
    // ---------------------------------------------------------------------------------------------
    if (IsGamma[is] && GammaHyp && is>=Nevents) {
        InvRmsE[is] = inv_rms_E(is);
    }

    //if (IsGamma[is] && GammaHyp || (!IsGamma[is] && !GammaHyp)) dxrec->Fill(currentX0-TrueX0[is]);
    //if (IsGamma[is] && GammaHyp || (!IsGamma[is] && !GammaHyp)) dyrec->Fill(currentY0-TrueY0[is]);

    //if (IsGamma[is] && GammaHyp || (!IsGamma[is] && !GammaHyp)) cout << is << " " << logL << " " << prevlogL << " " << istep << endl;
    return logL;
}



void FindLogLR_new (int is) {

    // Try delta method
    // ----------------
    // 1. find vector of gradients of the LLR versus all model parameters (5 for photons, 5 for protons)
    // 2. find inverse of hessian 10x10 matrix
    // 3. compute an estimate of the variance of the LLR by  variance = grad(LLR)^T * Hessian^-1 * grad(LLR) 
    //
    // The observable quantities for each detector id=1...Ndet are static floats:
    //   Nm[id][is], Ne[id][is], Tm[id][is], Te[id][is]
    // We have estimated parameters under the two hypotheses also as static doubles:
    //   double emg   = e_meas[is][0];
    //   double emp   = e_meas[is][1];
    //   double x0g   = x0meas[is][0];
    //   double x0p   = x0meas[is][1];
    //   double y0g   = y0meas[is][0];
    //   double y0p   = y0meas[is][1];
    //   double thmg  = thmeas[is][0];
    //   double thmp  = thmeas[is][1];
    //   double phmg  = phmeas[is][0];
    //   double phmp  = phmeas[is][1];
    // 
    // You can check in the routine FindShowerPos() how the likelihood is computed. There,  
    // the maximization is done by computing the first derivatives of the likelihood over the five
    // parameters. Try solving the problem by yourself first though, so that you double check my
    // calculations!
    // -----------------------------------------------------------------------------------------------------

    /*double conts[25];
    for (int i=0; i<5; i++)
    Double_t tmp[numberOfLines*numberOfColumns]; 
    for(int i=0;i<numberOfLines;i++) { 
        for(int j=0;j<numberOfColumns;j++) 
            tmp[i*numberOfColumns+j]=getValue(i,j); 
        }
    } 
    TMatrixD mat(numberOfLines,numberOfColumns,tmp); 
    mat.Invert(); //mat.InvertFast(); 
    if(!mat.IsValid()) return matrixNxM(0,0); 
    matrixNxM ret(numberOfColumns,numberOfLines); 
    for(int i=0;i<numberOfLines;i++) { 
        for(int j=0;j<numberOfColumns;j++) ret.setValue(i,j,mat[i][j]); 
    } return ret;

    Double_t det2;
    TMatrixD H2 = H_square;
    H2.Invert(&det2);

    TMatrixD U2(H2,TMatrixD::kMult,H_square);
    TMatrixDDiag diag2(U2); diag2 = 0.0;
    const Double_t U2_max_offdiag = (U2.Abs()).Max();
    std::cout << "  Maximum off-diagonal = " << U2_max_offdiag << std::endl;
    std::cout << "  Determinant          = " << det2 << std::endl;
    */
}


// Compute log likelihood ratio test statistic for one shower, by
// finding max value vs X0,Y0,theta,phi of shower for both hypotheses,
// and its variance. The variance is computed by sampling and with an
// analytic approximation, depending on SampleT. The routine also computes
// the derivatives of sigma2 over dx and dy
// NB 2/10/24 v133 now routine returns false if it runs in trouble, and
// event is skipped.
// -----------------------------------------------------------------------
bool FindLogLR (int is) { 

    // Compute the LRT
    // ---------------
    double logLG = FitShowerParams (is,true);  // Find shower position by max lik of gamma hypothesis
    double logLP = FitShowerParams (is,false); // Find shower position by max lik of proton hypothesis

    // The following has been commented out for now, as nothing gets discarded by 0.1/dof cut...
    // -----------------------------------------------------------------------------------------
    // If the shower can't be reconstructed reasonably for either hypothesis, we remove it from consideration
    // We set the threshold at p=10^-3 per detector and particle type. This makes the threshold pvalue be ptot < 10^(-6*Nunits)
    // ------------------------------------------------------------------------------------------------------------------------
    //double logpthresh = -6.*Nunits*log_10; // 6 factors per unit/id, log(0.1) each
    //cout << logpthresh << " " << logLG << " " << logLP << endl;
    //if (logLG<logpthresh && logLP<logpthresh) {
    //    Active[is] = false;
    //    sigmaLRT[is] = 100.; // or whatever
    //    return;
    //}

    logLRT[is] = logLG-logLP; 
    if (logLRT[is]!=logLRT[is]) {
        cout    << "Problem in FitShowerParams, logLG = " << logLG << " logLP = " << logLP << endl;
        outfile << "Problem in FitShowerParams, logLG = " << logLG << " logLP = " << logLP << endl;
        warnings1++;
        // TerminateAbnormally();
        return false;
    } else if (logLRT[is]>largenumber) {
        cout    << "Problem in FitShowerParams, logLG = " << logLG << " logLP = " << logLP << endl; 
        outfile << "Problem in FitShowerParams, logLG = " << logLG << " logLP = " << logLP << endl; 
        warnings4++;
        return false;
    }
    double sigmaT2_approx = 0.;
    double sigmaT2_sample = 0.; 
    double sigmaT2_deltam = 0.;

    if (SampleT) {
        int Nmstore[maxUnits];
        int Nestore[maxUnits];
        float Tmstore[maxUnits];
        float Testore[maxUnits];
        double X0st[2], Y0st[2], Thst[2], Phst[2], Est[2];
        X0st[0] = x0meas[is][0]; 
        Y0st[0] = y0meas[is][0]; 
        Thst[0] = thmeas[is][0];
        Phst[0] = phmeas[is][0];
        Est[0]  = e_meas[is][0];
        X0st[1] = x0meas[is][1]; 
        Y0st[1] = y0meas[is][1]; 
        Thst[1] = thmeas[is][1];
        Phst[1] = phmeas[is][1];
        Est[1]  = e_meas[is][1];

        // Store observed values
        // ---------------------
        for (int id=0; id<Nunits; id++) {
            Nmstore[id] = Nm[id][is];
            Nestore[id] = Ne[id][is];
            Tmstore[id] = Tm[id][is];
            Testore[id] = Te[id][is];
        }
        double SumProbGe1_store = SumProbGe1[is];
        bool Active_store       = Active[is];
        double PActive_store    = PActive[is];

        // Use repeated sampling to estimate the variance of T
        // ---------------------------------------------------
        double sumT  = 0.;
        double sumT2 = 0.;
        for (int irep=0; irep<Nrep; irep++) {
    
            // Regenerate counts and times
            // ---------------------------
            GetCounts (is); // This also modifies SumProbGe1, Active, and PActive, so we need to restore them at the end.
            double logLGi = FitShowerParams (is,true);  // Find shower position by max lik of gamma hypothesis, 1= fluctuate nm,ne,tm,te
            double logLPi = FitShowerParams (is,false); // Find shower position by max lik of proton hypothesis,1= fluctuate nm,ne,tm,te
            double T = logLGi-logLPi; 
            sumT += T; 
            sumT2+= T*T;
        }
        sigmaT2_sample = sumT2/Nrep - pow(sumT/Nrep,2);

        // Restore original observed values
        // --------------------------------
        for (int id=0; id<Nunits; id++) {
            Nm[id][is] = Nmstore[id];
            Ne[id][is] = Nestore[id];
            Tm[id][is] = Tmstore[id];
            Te[id][is] = Testore[id];
        }
        SumProbGe1[is] = SumProbGe1_store;
        Active[is]     = Active_store;
        PActive[is]    = PActive_store;

        x0meas[is][0] = X0st[0]; 
        y0meas[is][0] = Y0st[0]; 
        thmeas[is][0] = Thst[0];
        phmeas[is][0] = Phst[0];
        e_meas[is][0]  = Est[0];
        x0meas[is][1] = X0st[1]; 
        y0meas[is][1] = Y0st[1]; 
        thmeas[is][1] = Thst[1];
        phmeas[is][1] = Phst[1];
        e_meas[is][1]  = Est[1];
    } // End if sampleT
    if (sigmaT2_sample!=sigmaT2_sample) {
        cout    << "Warning, sigmaT2sample nan" << endl;
        outfile << "Warning, sigmaT2sample nan" << endl;    
        warnings5++;
    }
    // NB below we account for units with zero particles, which only get 2 dofs as the time measurements are non contributing
    if (sigmaT2_sample<=0.) sigmaT2_sample = pow(4.*Nunits_npgt0+2.*(Nunits-Nunits_npgt0),2.); // 4 measurements for Nunit detectors -> dchi2 = 4N per each L -> 2N + 2N
    
    // The rest is needed to compute an approximated formula for the variance of the LRT, and dsigmalrt_dx, dy values
    // --------------------------------------------------------------------------------------------------------------
    double emg   = e_meas[is][0];
    double emp   = e_meas[is][1];
    double x0g   = x0meas[is][0];
    double x0p   = x0meas[is][1];
    double y0g   = y0meas[is][0];
    double y0p   = y0meas[is][1];
    double thmg  = thmeas[is][0];
    double thmp  = thmeas[is][1];
    double phmg  = phmeas[is][0];
    double phmp  = phmeas[is][1];
    double et    = TrueE[is];
    double xt    = TrueX0[is];
    double yt    = TrueY0[is];
    double tt    = TrueTheta[is];
    double tp    = TruePhi[is];

    // If we write 
    //    LLR = sum_i {-lambda_mug + lambda_mup + N_mu [log(lambda_mug)-log(lambda_mup)] +
    //                 -0.5[(T_mu-T_expg)^2-(T_mu-T_expp)^2]/sigma_t^2 + (e terms)}
    // we can differentiate directly wrt lambdas and T_exps, obtaining
    //    s^2_LLR = sum_i [ (dLLR/dlambda_mug)^2 sigma^2(lambda_mug) + 
    //                      (dLLR/dlambda_mup)^2 sigma^2(lambda_mup) +
    //                      (dLLR/dNmu)^2 sigma^2 Nmu +                              <--- check if correct to add this. NOTE IT USES TRUE R
    //                      (dLLR/dTexpg)^2 sigma_T_expg^2 + 
    //                      (dLLR/dTexpp)^2 sigma_T_expp^2 + 
    //                      (dLLR/dTmu)^2 sigma_Tmeas^2 +                            <--- same
    //                      (e terms) ]
    // This becomes, as sigma^2(lambda) is -[d^2logL/dlambda^2]^-1 = lambda^2/N,
    //    s^2_LLR = sum_i { [(N_mu-lambda_mug)^2 + (N_mu-lambda_mup)^2] / N_mu + 
    //                      [(N_e-lambda_eg)^2 + (N_e-lambda_ep)^2] / N_e +          
    //                      [log(lambda_mug)-log(lambda_mup)]^2 * N_mu +             <--- check if correct to add this
    //                      [log(lambda_eg)-log(lambda_ep)]^2 * N_e +                <--- check if correct to add this
    //                      [(T_mu+T_e-2T_expg)^2 + (T_mu+T_e-2T_expp)^2] / sigma_texp^2 +
    //                      [-(T_mu+T_e-2T_expg)^2 -(T_mu+T_e-2T_expp)^2] / sigma_tmeas^2 }   <--- same
    //
    // Then, the derivative with respect to x[id] is found as follows:
    //  ds^2/dxi = [ -2(Nm-lmg)/Nm * dlmg/dRg * dRg/dxi -2(Nm-lmp)/Nm * dlmp/dRp * dRp/dxi +
    //               -2(Ne-leg)/Ne * dleg/drg * dRg/dxi -2(Ne-lep)/Ne * dlep/dRp * dRp/dxi +
    //               2Nm*(log(lmg)-log(lmp))*(1/lmg * dlmg/dRg * dRg/dxi + 1/lmp * dlmp/dRp * dRp/dxi) +
    //               2Ne*(log(leg)-log(lep))*(1/leg * dleg/dRg * dRg/dxi + 1/lep * dlep/dRp * dRp/dxi)] 
    //
    // and for now there are no contributions from time derivatives as they cancel, given that we have set equal sigma_time and sigma_texp.         
    // ------------------------------------------------------------------------------------------------------------------------------------
    for (int id=0; id<Nunits; id++) {        
        double xi     = x[id];
        double yi     = y[id];
        int nm        = Nm[id][is];
        int ne        = Ne[id][is];
        double thisRg = EffectiveDistance (xi,yi,x0g,y0g,thmg,phmg,0);
        double ctg    = cos(thmg);
        double lmg    = MFromG (emg, thmg, thisRg, 0) * ctg + fluxB_mu;
        double leg    = EFromG (emg, thmg, thisRg, 0) * ctg + fluxB_e;        
        if (nm>0.) sigmaT2_approx += pow(nm-lmg,2.)/nm;
        if (ne>0.) sigmaT2_approx += pow(ne-leg,2.)/ne;
        double thisRp = EffectiveDistance (xi,yi,x0p,y0p,thmp,phmp,0);
        double ctp    = cos(thmp);
        double lmp    = MFromP (emp, thmp, thisRp, 0) * ctp + fluxB_mu;
        double lep    = EFromP (emp, thmp, thisRp, 0) * ctp + fluxB_e;
        if (nm>0.) sigmaT2_approx += pow(nm-lmp,2.)/nm;
        if (ne>0.) sigmaT2_approx += pow(ne-lep,2.)/ne;

        // Take in the contribution from the variation of nm, ne. This requires us to compute
        // lambdas using the true shower parameters, not the ones that maximize L!
        // ----------------------------------------------------------------------------------
        double R    = EffectiveDistance (xi,yi,xt,yt,tt,tp,0);
        double ctt  = cos(tt);
        double lmgt = MFromG (et, tt, R, 0) * ctt + fluxB_mu;
        double legt = EFromG (et, tt, R, 0) * ctt + fluxB_e;
        double lmpt = MFromP (et, tt, R, 0) * ctt + fluxB_mu;
        double lept = EFromP (et, tt, R, 0) * ctt + fluxB_e;
        if (lmgt*lmpt>0.) sigmaT2_approx += pow(log(lmgt)-log(lmpt),2.)*nm;
        if (legt*lept>0.) sigmaT2_approx += pow(log(legt)-log(lept),2.)*ne;

        // Add contribution from time variation. This is commented out since
        // we have taken the two sigmas to be equal for now. 
        // -----------------------------------------------------------------
        /*
        double tm = Tm[id][is];
        double te = Te[id][is];
        double thisTg   = EffectiveTime(x[id],y[id],x00,y00,thm0,phm0,0);
        double thisTp   = EffectiveTime(x[id],y[id],x01,y01,thm1,phm1,0);
        sigmaT2_approx += pow(tm+te-2*thisTg,2)/sigma2_time;
        sigmaT2_approx += pow(tm+te-2*thisTp,2)/sigma2_time;
        sigmaT2_approx += -pow(tm+te-2*thisTg,2)/sigma2_texp; // verify where else this applies instead of sigma2_time!
        sigmaT2_approx += -pow(tm+te-2*thisTp,2)/sigma2_texp;
        */

        // Compute derivative of sigmaLRT^2 over dxi, dyi
        // ----------------------------------------------
        double dlmg_drg = MFromG (emg,thmg,thisRg,1) * ctg; // bgr part is constant
        double dlmp_drp = MFromP (emp,thmp,thisRp,1) * ctp;
        double dleg_drg = EFromG (emg,thmg,thisRg,1) * ctg;
        double dlep_drp = EFromP (emp,thmp,thisRp,1) * ctp;
        // And we need derivatives wrt R with true shower params, too.
        // ----------------------------------------------------------
        double dlmg_dr  = MFromG (et,tt,R,1) * ctt; // bgr part is constant
        double dlmp_dr  = MFromP (et,tt,R,1) * ctt;
        double dleg_dr  = EFromG (et,tt,R,1) * ctt;
        double dlep_dr  = EFromP (et,tt,R,1) * ctt;
        // And finally derivatives of radius true and measured vs x and y
        // --------------------------------------------------------------
        double dr_dxi   = -EffectiveDistance (xi,yi,xt,yt,tt,tp,1); // Note minus sign, as mode 1 is derivative wrt x0
        double dr_dyi   = -EffectiveDistance (xi,yi,xt,yt,tt,tp,2); // same, mode 2, y0
        double drg_dxi  = -EffectiveDistance (xi,yi,x0g,y0g,thmg,phmg,1);
        double drp_dxi  = -EffectiveDistance (xi,yi,x0p,y0p,thmp,phmp,1);
        double drg_dyi  = -EffectiveDistance (xi,yi,x0g,y0g,thmg,phmg,2);
        double drp_dyi  = -EffectiveDistance (xi,yi,x0p,y0p,thmp,phmp,2);

        double dsdx = 0.;
        double dsdy = 0.;
        if (nm>0.) {
            dsdx += -2.*(nm-lmg)/nm * dlmg_drg * drg_dxi 
                    -2.*(nm-lmp)/nm * dlmp_drp * drp_dxi; 
            dsdy += -2.*(nm-lmg)/nm * dlmg_drg * drg_dyi 
                    -2.*(nm-lmp)/nm * dlmp_drp * drp_dyi; 
            if (lmg*lmp>0.) { // Note, here we need the derivatives wrt the true R!
                double factor = 2.*nm*(log(lmg)-log(lmp))*(1./lmg * dlmg_dr + 1./lmp * dlmp_dr);
                dsdx += factor * dr_dxi;
                dsdy += factor * dr_dyi;
            } 
        }
        if (ne>0.) {
            dsdx += -2.*(ne-leg)/ne * dleg_drg * drg_dxi 
                    -2.*(ne-lep)/ne * dlep_drp * drp_dxi;
            dsdy += -2.*(ne-leg)/ne * dleg_drg * drg_dyi 
                    -2.*(ne-lep)/ne * dlep_drp * drp_dyi;
            if (leg*lep>0.) { // Note, here we need the derivatives wrt the true R!
                double factor = 2.*ne*(log(leg)-log(lep))*(1./leg * dleg_dr + 1./lep * dlep_dr);
                dsdx += factor * dr_dxi;
                dsdy += factor * dr_dyi;
            }
        }
        dsigma2_dx[id][is] = dsdx;
        dsigma2_dy[id][is] = dsdy;
    }
    if (sigmaT2_approx!=sigmaT2_approx) {
        cout    << "Warning, sigmaT2 nan" << endl;
        outfile << "Warning, sigmaT2 nan" << endl;
        warnings5++;
    }
    if (sigmaT2_approx<=0.) sigmaT2_approx = pow(4.*Nunits,2.);
    if (SampleT) {
        sigmaLRT[is] = sqrt(sigmaT2_sample);
    } else {
        sigmaLRT[is] = sqrt(sigmaT2_approx);
    }
    if (SampleT) {
        SvsS->Fill (log(sqrt(sigmaT2_approx)),log(sqrt(sigmaT2_sample)));
        SvsSP->Fill (log(sqrt(sigmaT2_approx)),log(sqrt(sigmaT2_sample)));
    }
    return true;
}

// Calculation of dlogLR over dx, dy
// ---------------------------------
std::pair<double,double> dlogLR_dxy (int id, int is) { 

    // We need to find a variation of T = logLg^max-logLp^max over the distance of shower is 
    // from a detector id. T varies if we change the distances, as expected fluxes vary with measured
    // distances. But T varies also because if we change the true distance from the shower
    // center the observed fluxes of particles change.
    // ----------------------------------------------------------------------------------------------
    double xi  = x[id];
    double yi  = y[id];
    double xmg = x0meas[is][0];
    double ymg = y0meas[is][0];
    double tmg = thmeas[is][0];
    double pmg = phmeas[is][0];
    double emg = e_meas[is][0];
    double xmp = x0meas[is][1];
    double ymp = y0meas[is][1];
    double tmp = thmeas[is][1];
    double pmp = phmeas[is][1];
    double emp = e_meas[is][1];
    double xt  = TrueX0[is];
    double yt  = TrueY0[is];
    double tt  = TrueTheta[is];
    double pt  = TruePhi[is];
    double et  = TrueE[is];
    double thisRg =  EffectiveDistance (xi,yi,xmg,ymg,tmg,pmg,0);
    double thisRp =  EffectiveDistance (xi,yi,xmp,ymp,tmp,pmp,0);
    double thisRt =  EffectiveDistance (xi,yi,xt,yt,tt,pt,0);
    double dRgdx  = -EffectiveDistance (xi,yi,xmg,ymg,tmg,pmg,1);
    double dRpdx  = -EffectiveDistance (xi,yi,xmp,ymp,tmp,pmp,1);
    double dRtdx  = -EffectiveDistance (xi,yi,xt,yt,tt,pt,1);
    double dRgdy  = -EffectiveDistance (xi,yi,xmg,ymg,tmg,pmg,2);
    double dRpdy  = -EffectiveDistance (xi,yi,xmp,ymp,tmp,pmp,2);
    double dRtdy  = -EffectiveDistance (xi,yi,xt,yt,tt,pt,2);
    double dx0 = xi - xmg;
    double dy0 = yi - ymg;
    double dx1 = xi - xmp;
    double dy1 = yi - ymp;
    double ct0 = cos(tmg);
    double ct1 = cos(tmp);
    double st0 = sin(tmg);
    double st1 = sin(tmp);
    double cp0 = cos(pmg);
    double cp1 = cos(pmp);
    double sp0 = sin(pmg);
    double sp1 = sin(pmp);
    double lambdaMG, lambdaEG;
    double lambdaMP, lambdaEP;

    // The calculation goes as follows, for an event k and detector unit i:
    // logLG = {-lambda_mug_i - lambda_eg_i + N_mug_i*log(lambda_mug_i) +
    //         N_eg_i*log(lambda_eg_i) -log(N_mu_i!) - log(N_e_i!) }
    // from which we get:
    // dlogLG/dR = { -dlambda_mug_i/dR_i - dlambda_eg_i/dR_i +d/dR_i(N_mug_i*log(lambda_mug_i)) +
    //             d/dR_i(N_eg_i*log(lambda_eg_i)) + d/dR_i(-log(N_mu_i!)) + d_dR_i(-log(N_mu_i!)) }
    // dlogLP/dR = { -dlambda_mup_i/dR_i - dlambda_ep_i/dR_i +d/dR_i(N_mup_i*log(lambda_mup_i)) +
    //             d/dR_i(N_ep_i*log(lambda_ep_i)) + d/dR_i(-log(N_mup_i!)) + d_dR_i(-log(N_mup_i!)) }
    //
    // Now, the factorials are derivatives with respect to true R, so they cancel in the logLR and we
    // ignore them. Instead, while the R_i deriving the lambdas are measured ones, the R_i deriving 
    // the N_ are the true ones. For these, we substitute N_ with the expectation values lambda at the 
    // same location. Further, since some of the factors depend on Rmeas and others on Rtrue, we find
    // it better to derive directly with respect to x[id], y[id].
    // We finally get the following expression:
    //    dlogLG/dx = dlambdag_mu/dRg_meas * dRg_meas/dx * (N_mu/lambdag_mu - 1) + 
    //                dlambdag_mu/dR_true dR_true/dx log(lambdag_mu) +             (note N->lambda here)
    //                dlambdag_e/dRg_meas * dRg_meas/dx * (N_e/lambdag_e - 1) +
    //                dlambdag_e/dR_true dR_true/dx log(lambdag_e)                 (and here)
    // and similarly for the protons.
    // As far as time dependence goes, we note that Tmeas changes with Rtrue, and Texp with Rmeas,
    // so we need to account for both and 
    //    dlogLG/dx += -(Tmeas-Texp)/sigma^2 * dTmeas/dRtrue * dRtrue/dx +
    //                 +(Tmeas-Texp)/sigma^2 * dTexp/dRmeas  * dRmeas/dx      
    // -----------------------------------------------------------------------------------------------

    // These are fluxes from measured shower coords
    // --------------------------------------------
    lambdaMG = MFromG (emg, tmg, thisRg, 0)*ct0 + fluxB_mu;
    lambdaEG = EFromG (emg, tmg, thisRg, 0)*ct0 + fluxB_e;
    lambdaMP = MFromP (emp, tmp, thisRp, 0)*ct1 + fluxB_mu;
    lambdaEP = EFromP (emp, tmp, thisRp, 0)*ct1 + fluxB_e;
    double fbgm = fluxB_mu/lambdaMG; // Unfortunately here we have to diversify the fractions as they affect the hypotheses 
    double fbge = fluxB_e/lambdaEG;
    double fbpm = fluxB_mu/lambdaMP;
    double fbpe = fluxB_e/lambdaEP;
    double thisTgm = 0.;
    double thisTpm = 0.;
    double thisTge = 0.;
    double thisTpe = 0.;
    if (!OrthoShowers) { 
        double tg = EffectiveTime (xi,yi,xmg,ymg,tmg,pmg,0);
        double tp = EffectiveTime (xi,yi,xmp,ymp,tmp,pmp,0);
        thisTgm = tg; // (1.-fbgm)*tg; for now we implement no bgr dep on time
        thisTpm = tp; // (1.-fbpm)*tp;
        thisTge = tg; // (1.-fbge)*tg;
        thisTpe = tp; // (1.-fbpe)*tp;
    }

    // Derivatives with respect to Rmeas
    // ---------------------------------
    double dlMGdR, dlEGdR;
    double dlMPdR, dlEPdR;
    dlMGdR = MFromG (emg, tmg, thisRg, 1) * ct0;
    dlEGdR = EFromG (emg, tmg, thisRg, 1) * ct0;
    dlMPdR = MFromP (emp, tmp, thisRp, 1) * ct1;
    dlEPdR = EFromP (emp, tmp, thisRp, 1) * ct1;

    // Now assemble the pieces
    // -----------------------
    double dlogLGdR = -dlMGdR-dlEGdR; // This is the contribution of dlambda/dRmeas values
    double dlogLPdR = -dlMPdR-dlEPdR;
    int nm         = Nm[id][is];
    int ne         = Ne[id][is];
    float tm       = Tm[id][is];
    float te       = Te[id][is];
    if (lambdaMG>0.) {
        dlogLGdR += nm/lambdaMG * dlMGdR; 
    }
    if (lambdaEG>0.) {
        dlogLGdR += ne/lambdaEG * dlEGdR;
    }
    if (lambdaMP>0.) {
        dlogLPdR += nm/lambdaMP * dlMPdR; 
    }
    if (lambdaEP>0.) {
        dlogLPdR += ne/lambdaEP * dlEPdR; 
    }

    // Now the result of above sums has to be multiplied by dR_meas/dx or dy
    // ---------------------------------------------------------------------
    double dlogLGdx, dlogLGdy, dlogLPdx, dlogLPdy;
    dlogLGdx = dlogLGdR * dRgdx;
    dlogLPdx = dlogLPdR * dRpdx; 
    dlogLGdy = dlogLGdR * dRgdy;
    dlogLPdy = dlogLPdR * dRpdy;

    // We now add the contribution log(lambda_mu) * dN_mu/dR_true * dR_true/dx (dy), and e term.
    // Here we take N_mu == lambda_mu to get its R_true derivative.
    // -----------------------------------------------------------------------------------------
    double factor = cos(tt)*dRtdx;
    if (lambdaMG>0.) dlogLGdx += MFromG (et,tt,thisRt,1) * factor * log(lambdaMG); 
    if (lambdaEG>0.) dlogLGdx += EFromG (et,tt,thisRt,1) * factor * log(lambdaEG); 
    if (lambdaMP>0.) dlogLPdx += MFromP (et,tt,thisRt,1) * factor * log(lambdaMP); 
    if (lambdaEP>0.) dlogLPdx += EFromP (et,tt,thisRt,1) * factor * log(lambdaEP); 
    factor = cos(tt)*dRtdy;
    if (lambdaMG>0.) dlogLGdy += MFromG (et,tt,thisRt,1) * factor * log(lambdaMG); 
    if (lambdaEG>0.) dlogLGdy += EFromG (et,tt,thisRt,1) * factor * log(lambdaEG); 
    if (lambdaMP>0.) dlogLPdy += MFromP (et,tt,thisRt,1) * factor * log(lambdaMP); 
    if (lambdaEP>0.) dlogLPdy += EFromP (et,tt,thisRt,1) * factor * log(lambdaEP); 

    // Handle variation of logL on R due to variation of thisT
    // -------------------------------------------------------
    if (!OrthoShowers) { 
        double t0      = dx0*st0*cp0 + dy0*st0*sp0;
        double t1      = dx1*st1*cp1 + dy1*st1*sp1;
        double den01   = dx0-t0*st0*cp0;
        double den02   = dy0-t0*st0*sp0;
        double den11   = dx1-t1*st1*cp1;
        double den12   = dy1-t1*st1*sp1;
        double dTg_dRi = maxdTdR;
        if (den01!=0. && den02!=0.) dTg_dRi = thisRg/c0 * ( st0*cp0/den01 + st0*sp0/den02 );
        double dTp_dRi = maxdTdR;
        if (den11!=0. && den12!=0.) dTp_dRi = thisRp/c0 * ( st1*cp1/den11 + st1*sp1/den12 );
        double dTtrue_dxtrue = EffectiveTime (xi,yi,xt,yt,tt,pt,1);     // This returns -sintht cospht /c
        double dTg_dxmeas    = EffectiveTime (xi,yi,xmg,ymg,tmg,pmg,1); // This instead -sinthmg*cosphmg /c
        double dTp_dxmeas    = EffectiveTime (xi,yi,xmp,ymp,tmp,pmp,1);
        // Both true and measured time depend on xi
        if (nm>0) {
            dlogLGdx +=  (tm-thisTgm)/sigma2_time * (dTg_dxmeas-dTtrue_dxtrue);
            dlogLPdx +=  (tm-thisTpm)/sigma2_time * (dTp_dxmeas-dTtrue_dxtrue);
        } 
        if (ne>0) {
            dlogLGdx +=  (te-thisTge)/sigma2_time * (dTg_dxmeas-dTtrue_dxtrue);
            dlogLPdx +=  (te-thisTpe)/sigma2_time * (dTp_dxmeas-dTtrue_dxtrue);
        }
        double dTtrue_dytrue = EffectiveTime (xi,yi,xt,yt,tt,pt,2);
        double dTg_dymeas    = EffectiveTime (xi,yi,xmg,ymg,tmg,pmg,2);
        double dTp_dymeas    = EffectiveTime (xi,yi,xmp,ymp,tmp,pmp,2);
        if (nm>0) {
            dlogLGdy +=  (tm-thisTgm)/sigma2_time * (dTg_dymeas-dTtrue_dytrue);
            dlogLPdy +=  (tm-thisTpm)/sigma2_time * (dTp_dymeas-dTtrue_dytrue);
        }
        if (ne>0) {
            dlogLGdy +=  (te-thisTge)/sigma2_time * (dTg_dymeas-dTtrue_dytrue);
            dlogLPdy +=  (te-thisTpe)/sigma2_time * (dTp_dymeas-dTtrue_dytrue);
        }
    } // End if !OrthoShowers

    if (dlogLGdR!=dlogLGdR || dlogLPdR!=dlogLPdR) {
        cout    << "Warning - Trouble in dlogLdR" << endl;
        outfile << "Warning - Trouble in dlogLdR" << endl;
        cout    << thisTge << " " << thisTpe << " " << sigma2_time << endl; //  << " " << dTg_dRi << " " << dTp_dRi << endl;
        cout    << thisRg << " " << thisRp << " " << dlMGdR << " " << dlEGdR << " " << dlMPdR << " " << dlEPdR << endl;
        cout    << lambdaMG << " " << lambdaEG << " " << lambdaMP << " " << lambdaEP << endl << endl; 
        outfile << thisTge << " " << thisTpe << " " << sigma2_time << endl; //  << " " << dTg_dRi << " " << dTp_dRi << endl;
        outfile << thisRg << " " << thisRp << " " << dlMGdR << " " << dlEGdR << " " << dlMPdR << " " << dlEPdR << endl;
        outfile << lambdaMG << " " << lambdaEG << " " << lambdaMP << " " << lambdaEP << endl << endl; 
        SaveLayout();
#if defined(STANDALONE) || defined(UBUNTU)
        datamutex.lock();
#endif
        warnings3++; 
#if defined(STANDALONE) || defined(UBUNTU)
        datamutex.unlock();
#endif
        return std::make_pair(0.,0.);
    }
    // Return function value
    // ---------------------
    //cout << "id, is = " << id << " " << is << " " << dlogLGdx-dlogLPdx << " " << dlogLGdy-dlogLPdy << endl;
    return std::make_pair(dlogLGdx-dlogLPdx,dlogLGdy-dlogLPdy);
}

// This routine computes the PDF of the test statistic for a primary hypothesis
// ----------------------------------------------------------------------------
double ComputePDF (int k, bool gammahyp) {

    // We introduce the calculation of the probability that at least N>=Ntrigger
    // detectors observe >0 particles (muons plus (ele+gamma)) and incorporate it
    // into the probability that the shower is a gamma or a proton, as the event
    // will be in the considered sample only if N>=Ntrigger units saw a signal.
    // This is computed by taking a Poisson approximation, as follows:
    // Each detector has an expectation value for the number of observed particles
    // that is equal to 
    //     xi_i = lambda_mu(i)+lambda_e(i). 
    // This depends of course on the hypothesis for the considered shower. We have
    //     P_i(>=1 particle) = 1 - exp(-xi_i)
    // and we form the sum of these P_i for all detectors:
    //     S = sum_i P_i
    // We then get the probability that the event passes the trigger, and is thus
    // included in the sample, as the approximated value
    //     Pactive[is] = 1 - sum_{k=0}^{Ntrigger-1} Poisson(k|S)
    // This is checked to be a good representation of the true probability by the routine
    // at the end of this macro (CheckProb).
    // ----------------------------------------------------------------------------------
    double p     = 0.;
    double Norm  = 0.;
    for (int m=0; m<Nevents; m++) {
        if (!Active[m] || IsGamma[m] !=gammahyp) continue;
        double sigma = sigmaLRT[m];
        double Gden  = sqrt2pi*sigma; 
        double G     = PActive[m]/Gden*exp(-pow((logLRT[m]-logLRT[k])/sigma,2.)/2.);
        if (G!=G) {
            cout    << "Warning, NANs in PDF calculations - sigmaLRT[" << m << "] = " << sigmaLRT[m] 
                    << " logLRT[" << m << "] = " << logLRT[m] << " logLRT[" << k << "] = " << logLRT[k] << endl;
            outfile << "Warning, NANs in PDF calculations - sigmaLRT[" << m << "] = " << sigmaLRT[m] 
                    << " logLRT[" << m << "] = " << logLRT[m] << " logLRT[" << k << "] = " << logLRT[k] << endl;
            warnings4++;
            continue; 
        }
        if (G>epsilon2) {
            p += G; // Otherwise it screws around with E-170 numbers and returns bogus results
        }
        Norm += PActive[m]; // Also zeros contribute to denominator        
    }
    // We have summed Gaussians (already normalized) an effective number of times equal to Ng or Np,
    // so to normalize the pdfs we need to divide by those numbers. This includes PActive[k] terms.
    // --------------------------------------------------------------------------------------------- 
    if (Norm>0.) p /= Norm; 
    if (p<epsilon2) {
        p = epsilon2; // Protect against outliers screwing up calculations
    }
    return p;
}

// Compute variation of dlogL_dE for small variation of E. This is used to extract
// an estimate of dEk/dRik for shower k and detector i, in the calculation of the
// derivative of the integrated resolution, for the Utility calculation.
// Because it is too CPU costly to re-maximize the logLikelihood for a variation
// of detector position, we compute delta(dlogL/dE)/deltaE and delta(dlogL/dE)/deltaR,
// and force them to be equal and of opposite sign. The routine below computes the
// first part. It is called by the other routine, which determines dE_dR altogether.
// 
// Note, here we are only concerned with energy estimates under the gamma hypothesis!
// ---------------------------------------------------------------------------------------------
double delta_dlogLdE (int is, double de) {
    double ethis        = e_meas[is][0];
    double eprime       = ethis + de; // We are not exceeding bounds as if ethis is close to 10 PeV de is made negative in calling this function; however it is a bit of a kludge, beware
    double dlogLdE      = 0.;
    double dlogLdEprime = 0.;
    double tm           = thmeas[is][0];
    double ct           = cos(tm);
    for (int id=0; id<Nunits; id++) {
        int nm        = Nm[id][is];
        int ne        = Ne[id][is];
        double thisR    = EffectiveDistance (x[id],y[id],x0meas[is][0],y0meas[is][0],tm,phmeas[is][0],0);          
        double lambdaM  = MFromG (ethis, tm, thisR, 0)*ct + fluxB_mu;
        double lambdaE  = EFromG (ethis, tm, thisR, 0)*ct + fluxB_e;
        double dlMdE    = MFromG (ethis, tm, thisR, 2)*ct;
        double dlEdE    = EFromG (ethis, tm, thisR, 2)*ct;
        dlogLdE        -= (dlMdE + dlEdE);
        if (lambdaM>0.) {
            dlogLdE += nm/lambdaM * dlMdE; 
        }
        if (lambdaE>0.)  {
            dlogLdE += ne/lambdaE * dlEdE;
        }
        // Now for modified energy, same detector positions
        // ------------------------------------------------
        lambdaM         = MFromG (eprime, tm, thisR, 0)*ct + fluxB_mu;
        lambdaE         = EFromG (eprime, tm, thisR, 0)*ct + fluxB_e;
        dlMdE           = MFromG (eprime, tm, thisR, 2)*ct;
        dlEdE           = EFromG (eprime, tm, thisR, 2)*ct;
        dlogLdEprime   -= (dlMdE + dlEdE);
        if (lambdaM>0.) {
            dlogLdEprime += nm/lambdaM * dlMdE; 
        }
        if (lambdaE>0.)  {
            dlogLdEprime += ne/lambdaE * dlEdE;
        }
    }
    // cout << " E reco: dlogLdE', dlogLdE = " << dlogLdEprime << " " << dlogLdE << " Em,Et = " << e_meas[is][0] << "," << TrueE[is] << endl;
    return dlogLdEprime - dlogLdE; 
}

// Compute derivative of measured energy over measured R (shower, detector) 
// by the procedure explained above (See routine delta_dlogLdE)
// ------------------------------------------------------------------------                     
double dEk_dRik_old (int id, int is) {

    // Perform actual calculation with new likelihood maximization
    // -----------------------------------------------------------
    double de_real;
    double previousE = e_meas[is][0];
    double previousx = x0meas[is][0];
    double previousy = y0meas[is][0];
    double previoust = thmeas[is][0];
    double previousp = phmeas[is][0];
    double newincr   = 1.; // 0.00000001*DetectorSpacing;
    double thisR     = EffectiveDistance (x[id],y[id],previousx,previousy,previoust,previousp,0);
    double dx = x[id]-previousx; // Delta x on the ground from core
    double dy = y[id]-previousy; // Delta y on the ground from core
    double dr = sqrt(dx*dx+dy*dy); 
    if (dedrtrue) {
        // Modify this detector position to redo the likelihood fit
        // --------------------------------------------------------
        x[id] = x[id] + newincr*dx/dr; // This movement increases the true distance from the axis 
        y[id] = y[id] + newincr*dy/dr; // ... by newincr*dr/thisR
        FitShowerParams(is,true);
        de_real = e_meas[is][0]-previousE; // This is E'_hat - E_hat, i.e. variation in E due to operated var in R 
        x[id] = x[id] - newincr*dx/dr; // Restore correct x of detector position
        y[id] = y[id] - newincr*dy/dr; // ... and correct y
        // Restore measured shower parameters as they were before we tentatively moved detector id
        // ---------------------------------------------------------------------------------------
        e_meas[is][0] = previousE;
        x0meas[is][0] = previousx;
        y0meas[is][0] = previousy;
        thmeas[is][0] = previoust;
        phmeas[is][0] = previousp;
        cout << "  Direct calc = " << de_real/(newincr*thisR/dr) << " ";
        //return de_real/(newincr*thisR/dr);
    } else {
        // Compute variation in dlogL/dE for tiny E change
        // -----------------------------------------------
        double de = 0.0001*TrueE[is]; // de_real; // 0.00000000001; // 0.000000001*previousE;
        if (previousE>=Emax-de) de = -de; // Only consider downward shifts if E is at max boundary
                                          // ... otherwise if previousE is at min boundary, a positive shift is ok
        double deltaf_E = delta_dlogLdE (is,de); // This is dlogLdE(E+de) - dlogLdE(E)
        
        // Now we search for the dR that produces the opposite shift in dlogLdE
        // --------------------------------------------------------------------
        int nm         = Nm[id][is];
        int ne         = Ne[id][is];
        float tm         = Tm[id][is];
        float te         = Te[id][is]; 
        double ct        = cos(previoust);
        double lambdaM   = MFromG (previousE, previoust, thisR, 0)*ct + fluxB_mu;
        double lambdaE   = EFromG (previousE, previoust, thisR, 0)*ct + fluxB_e;
        double dlMdE     = MFromG (previousE, previoust, thisR, 2)*ct;
        double dlEdE     = EFromG (previousE, previoust, thisR, 2)*ct;
        double dlogLdE   = -(dlMdE + dlEdE);
        if (lambdaM>0.) {
            dlogLdE += nm/lambdaM * dlMdE; 
        }
        if (lambdaE>0.)  {
            dlogLdE += ne/lambdaE * dlEdE;
        }
        
        // Now compute modified dlogLdE, iterating
        // ---------------------------------------
        int iloop           = 0;
        int maxloops        = 100;
        double distmax      = 0.00000000001;
        double dist         = 0.;
        double olddist      = largenumber;
        double maxincrement = 2.*ArrayRspan[0];
        double increment    = 1.; // Test radial displacement - 0001*DetectorSpacing;
        double oldincrement = 0.;
        double thisRprime   = thisR + increment;
        //emeas               = emeas+de;
        double dlogLdEprime = 0.;
        do {
            if (thisRprime<Rmin) thisRprime = Rmin; // Protect it
            lambdaM            = MFromG (previousE, previoust, thisRprime, 0)*ct + fluxB_mu;
            lambdaE            = EFromG (previousE, previoust, thisRprime, 0)*ct + fluxB_e;
            dlMdE              = MFromG (previousE, previoust, thisRprime, 2)*ct;
            dlEdE              = EFromG (previousE, previoust, thisRprime, 2)*ct;
            dlogLdEprime       = -(dlMdE + dlEdE);
            if (lambdaM>0.) {
                dlogLdEprime += nm/lambdaM * dlMdE; 
            }
            if (lambdaE>0.)  {
                dlogLdEprime += ne/lambdaE * dlEdE;
            }

            // Now we compute the sum of the variation in dlnL/dE caused by E'=E+de
            // and the variation in dlnL/dE caused by R'=R+dr. The rationale is that dlnL/dE must
            // stay zero for the MLE solution E_hat, so if R varies to R', then E_hat must vary to E'_hat
            // so that E'_hat is the MLE at R'. This allows to compute dE_hat/dR = (E'_hat-E_hat)/(R'-R).
            // We computed earlier dlnL/dE(E')-dlnL/dE(E) = deltaf_E, now we also have
            // dlnL/dE(R') - dlnL/dE(R), and we force the two variations to be opposite by adjusting R'.
            // ------------------------------------------------------------------------------------------
            dist = deltaf_E + (dlogLdEprime-dlogLdE);
            // cout << " il = " << iloop << " dist = " << dist << "old dist = " << olddist << " incr, old = " << increment << " " << oldincrement << endl;
            if (fabs(dist)<fabs(distmax)) continue; // We've done it
            if (iloop>0) {
                double R = fabs(dist)/fabs(olddist);
                if (dist*olddist>0.) { // On same side of zero
                    if (R<1.) { // We reduced it in size
                        double tmp = oldincrement;
                        oldincrement = increment;
                        increment += R/(1.-R)*(increment-tmp); // Linear approximation for increment
                        if (fabs(increment)>maxincrement) {
                            if (increment>0.) {
                                increment = maxincrement;
                            } else {
                                increment = -maxincrement;
                            }
                        }
                    } else if (R>1.) { // We increased it, go back
                        double tmp = oldincrement;
                        oldincrement = increment;
                        increment = tmp +0.5*(tmp-increment); // Jump on the other side
                    } else {
                        oldincrement = increment;
                        increment *= 1.2;
                    }
                } else if (dist*olddist<0.) {
                    if (R<1.) {
                        double tmp = oldincrement;
                        oldincrement = increment;
                        increment = increment +1./(1.+1./R)*(tmp-increment); // Linear approx
                        if (fabs(increment)>maxincrement) {
                            if (increment>0.) {
                                increment = maxincrement;
                            } else {
                                increment = -maxincrement;
                            }
                        }
                    } else if (fabs(dist)>fabs(olddist)) {
                        double tmp = increment;
                        oldincrement = increment;
                        increment = tmp + 0.5*(tmp-increment); // Jump on other side
                    } else {
                        oldincrement = increment;
                        increment *= 0.8;
                    }
                }
            } else {
                oldincrement = increment;
                increment = 2.*increment;
            }
            thisRprime = thisR + increment;
            if (thisRprime<0.) thisRprime = 0.;
            olddist = dist;
            iloop++;
        } while (fabs(dist)>fabs(distmax) && iloop<maxloops); // && increment<4.*ArrayRspan);
        //cout << "     N loops = " << iloop << " dist = " << dist << " E, Et = " << e_meas[is][0] << " " << TrueE[is] 
        //     << " increment = " << increment;
        //cout << " deltaR = " << thisRprime-thisR << endl;
        //cout << "df = " << deltaf_E << " dlogLdeprime = " << dlogLdEprime << " dlogLde = " << dlogLdE << endl;

        if (thisRprime-thisR!=0.) cout << " " << de/(thisRprime-thisR) << " " << iloop << endl;

        //if (de!=0.) {
        //    if (fabs(de_real/(newincr*thisR/dr)/(de/(thisRprime-thisR))-1.)<0.1) sumrat++;
        //    nsum++;
        //}
        if (fabs(dist)<fabs(distmax) && thisRprime!=thisR) {
            //cout << "nloops = " << iloop << " dedr, de, incr = " << de/increment << " " << de << " " << increment << " dist = " << dist << " xd,yd = " 
            //        << x[id] << "," << y[id] 
            //        << " x0,y0m = " << x0meas[is][0] << " " << y0meas[is][0] << " Et, Em = " << TrueE[is] << " " << e_meas[is][0] << endl; 
            //return -de/increment; 
            return de/(thisRprime-thisR);
        } else {
            //cout << " returning 0, dist = " << dist << " nloops = " << iloop << " xd,yd = " 
            //        << x[id] << "," << y[id] << " x0,y0m = " << x0meas[is][0] << " " << y0meas[is][0] << " Et, Em = " 
            //        << TrueE[is] << " " << e_meas[is][0] << endl; 
            return 0.; // failure
        }
    } // endif actual calc
    return 0.;
}

// Compute derivative of measured energy over measured R (shower, detector) 
// by the procedure explained above (See routine delta_dlogLdE)
// ------------------------------------------------------------------------                     
double dEk_dRik (int istar, int k) {

    //dEk_dRik_old(istar,k); // check actual calculation of variation

    // We compute dEmeas/dRmeas as - d2logL/dedr / d2logL/de2
    // (this follows from f(x,y)=0, f(x+dx,y+dy)=0 --> dy = -df/dx / df/dy * dx ->  dx/dy = - df/dy / df/dx )
    // with x=e, y=r, f = dlogL/de 
    // ------------------------------------------------------------------------------------------------------
    double em        = e_meas[k][0];
    double tm        = thmeas[k][0];
    double xm        = x0meas[k][0];
    double ym        = y0meas[k][0];
    double pm        = phmeas[k][0];
    double ct        = cos(tm);
    double dlogLdE   = 0.;
    double d2logLdE2 = 0.;
    int nm, ne;
    double R, lambdaM, lambdaE, dlMdE, dlEdE, d2lMdE2, d2lEdE2;
    for (int id=0; id<Nunits; id++) {
        nm       = Nm[id][k];
        ne       = Ne[id][k];
        R        = EffectiveDistance (x[id],y[id],xm,ym,tm,pm,0);
        lambdaM  = MFromG (em, tm, R, 0)*ct + fluxB_mu;
        lambdaE  = EFromG (em, tm, R, 0)*ct + fluxB_e;
        dlMdE    = MFromG (em, tm, R, 2)*ct;
        dlEdE    = EFromG (em, tm, R, 2)*ct;
        dlogLdE += -(dlMdE + dlEdE);
        if (lambdaM>0.) dlogLdE += nm/lambdaM * dlMdE;         
        if (lambdaE>0.) dlogLdE += ne/lambdaE * dlEdE;
        d2lMdE2    = MFromG (em, tm, R, 22)*ct;
        d2lEdE2    = EFromG (em, tm, R, 22)*ct;
        d2logLdE2 += -d2lMdE2 + nm/lambdaM*(-1./lambdaM*pow(dlMdE,2.) + d2lMdE2) 
                     -d2lEdE2 + ne/lambdaE*(-1./lambdaE*pow(dlEdE,2.) + d2lEdE2);
    }
    R                 = EffectiveDistance (x[istar],y[istar],xm,ym,tm,pm,0);
    nm                = Nm[istar][k];
    ne                = Ne[istar][k];
    double d2lMdEdR   = MFromG (em, tm, R, 23)*ct; // Variation wrt measured E and wrt radius core to detector
    double d2lEdEdR   = EFromG (em, tm, R, 23)*ct;
    double dlMdR      = MFromG (em, tm, R,  1)*ct;
    double dlEdR      = EFromG (em, tm, R,  1)*ct;
    lambdaM           = MFromG (em, tm, R,  0)*ct + fluxB_mu;
    lambdaE           = EFromG (em, tm, R,  0)*ct + fluxB_e;
    dlMdE             = MFromG (em, tm, R,  2)*ct;
    dlEdE             = EFromG (em, tm, R,  2)*ct;
    double d2logLdEdR = -d2lMdEdR + nm/lambdaM*(-1./lambdaM*dlMdR*dlMdE + d2lMdEdR) 
                        -d2lEdEdR + ne/lambdaE*(-1./lambdaE*dlEdR*dlEdE + d2lEdEdR);
    if (d2logLdE2!=0.) {
        //cout << ", sec der. ratio = " << -d2logLdEdR/d2logLdE2 << endl;
        return -d2logLdEdR/d2logLdE2;
    } 
    return 0.;
}

// Function to sort points by polar angle with respect to the first point
// ----------------------------------------------------------------------
bool compare (const Point& p1, const Point& p2, const Point& base) {
    double crossProduct = (p1.yy - base.yy) * (p2.xx - base.xx) - (p1.xx - base.xx) * (p2.yy - base.yy);
    if (crossProduct==0) {
        // If points are collinear, prioritize the one closest to the base point
        // ---------------------------------------------------------------------
        return (p1.xx - base.xx) * (p1.xx - base.xx) + (p1.yy - base.yy) * (p1.yy - base.yy) < 
               (p2.xx - base.xx) * (p2.xx - base.xx) + (p2.yy - base.yy) * (p2.yy - base.yy);
    }
    return crossProduct>0; // Sort in counterclockwise order
}

double CrossProduct (const Point& p1, const Point& p2, const Point& base) {
    return (p1.yy-base.yy)*(p2.xx-base.xx) - (p1.xx-base.xx)*(p2.yy-base.yy);
}

// Comparison function to sort points by their cross product with a base point
// ---------------------------------------------------------------------------
bool compareByCrossProduct (const Point& a, const Point& b, const Point& base) {
    return CrossProduct(a, b, base) < 0; // Change '<' to '>' if needed
}

bool compareByAngle(const Point& p1, const Point& p2, const Point& base) {
    double angle1 = atan2(p1.yy - base.yy, p1.xx - base.xx);
    double angle2 = atan2(p2.yy - base.yy, p2.xx - base.xx);
    return angle1 < angle2; // Sort by increasing angle
}

double Angle (const Point& p1, const Point& base) {
    return atan2(p1.yy - base.yy, p1.xx - base.xx);
}

double computeAngle (const Point& A, const Point& B, const Point& C) {
    // Vector BA
    double BAx = A.xx - B.xx;
    double BAy = A.yy - B.yy;

    // Vector BC
    double BCx = C.xx - B.xx;
    double BCy = C.yy - B.yy;

    // Dot product BA · BC
    double dotProduct = BAx * BCx + BAy * BCy;

    // Magnitudes |BA| and |BC|
    double magnitudeBA = std::sqrt(BAx * BAx + BAy * BAy);
    double magnitudeBC = std::sqrt(BCx * BCx + BCy * BCy);

    // Cosine of the angle
    double cosTheta = dotProduct / (magnitudeBA * magnitudeBC);

    // Calculate the angle
    return std::acos(cosTheta);
}

// Area of array. Compute as area of Convex Hull enclosing all detector units
// We base the calculus on the x[id], y[id] positions, but if idmod is >=0
// we change the position of idmod, to compute the derivative by interpolation
// (see code ComputeUtilityArea())
// ---------------------------------------------------------------------------
double ConvexHull (int idmod=-1, double dx=0., double dy=0., bool print=false) {

    std::vector<Point> points;
    for (int i=0; i<Nunits; i++) { 
        Point p;   
        p.xx = x[i];
        p.yy = y[i];
        if (i==idmod) p.xx += dx;
        if (i==idmod) p.yy += dy;
        points.push_back(p);
    }

    // Step 1: Find the point with the lowest y-coordinate (and lowest x-coordinate if tied)
    // -------------------------------------------------------------------------------------
    int minIndex = 0;
    for (int i = 1; i<points.size(); ++i) { 
        if (points[i].yy<points[minIndex].yy || 
            (points[i].yy==points[minIndex].yy && points[i].xx<points[minIndex].xx)) {
            minIndex = i;
        }
    }
    std::swap (points[0], points[minIndex]);
    // cout << "Points 0 swapped" << endl;
    Point base = points[0];

    // Step 2: Sort the points based on polar angle with respect to the first point
    // ----------------------------------------------------------------------------
    std::sort (points.begin()+1, points.end(), [&](const Point& a, const Point& b) { 
            return compareByAngle(a,b,base); // compareByCrossProduct (a,b,base);
        }
    );
    //    for (int i=0; i<points.size(); i++) {
    //        cout << i << ": (" << points[i].xx << ", " << points[i].yy << ") angle = ";
    //        if (i>0) cout << atan2(points[i].yy-points[0].yy,points[i].xx-points[0].xx) << endl;
    //    }
    //    cout << endl;

    // Step 3: Initialize stack for storing the convex hull vertices
    // -------------------------------------------------------------
    std::stack<Point> convexHull;

    // Step 4: Push first three points to the stack
    // --------------------------------------------
    convexHull.push(points[0]);
    convexHull.push(points[1]);

    // Step 5: Process remaining points
    // --------------------------------
    for (int i=2; i<points.size(); ++i) { // Nunits; ++i) {
        Point top = convexHull.top();
        convexHull.pop(); // Temporarily remove the top point
        Point next_to_top = convexHull.top();
        int maxangleind = -1;
        double maxangle = -largenumber;
        for (int j=i; j<points.size(); ++j) {
            double thisangle = computeAngle(next_to_top,top,points[j]);
            if (thisangle>maxangle) {
                maxangle = thisangle;
                maxangleind = j;
            }
            //cout << j << " point: " << points[j].xx << " " << points[j].yy 
            //     << " " << thisangle << " " << maxangle << " " << maxangleind << endl;
        }
        //cout << " choose " << maxangleind << endl;
        
        if (maxangleind>-1) {
            i = maxangleind; // We will not further consider points that had indexes below j
            convexHull.push(top); // If it's valid, put top point back
            convexHull.push(points[i]); // Now push the current point onto the stack
        } else {
            convexHull.push(top); // We ignore points[i] as it makes a clockwise turn
            i = points.size(); // We exit the loop 
        }
    }

    // Convert the stack into a vector of points (otherwise can't access i-th element)
    // -------------------------------------------------------------------------------
    std::vector<Point> hullVector;
    while (!convexHull.empty()) {
        hullVector.push_back(convexHull.top());
        convexHull.pop();
    }
    std::reverse(hullVector.begin(), hullVector.end()); 
    
    if (print) {
        cout << "      Size of hull = " << hullVector.size() << endl;
        for (int i=0; i<hullVector.size(); i++) {
            cout << "     i = " << i << "x,y = " << hullVector[i].xx << "," << hullVector[i].yy << endl;
        }
        cout << endl;
    }

    // Compute the signed area of the polygon using the shoelace formula
    // -----------------------------------------------------------------
    double Area = 0.;
    for (size_t i=0; i<hullVector.size(); ++i) {
        size_t j=(i+1)%hullVector.size(); // Next point index
        Area += hullVector[i].xx*hullVector[j].yy - hullVector[j].xx*hullVector[i].yy;
    }
    if (Area!=Area) {
        cout    << "ConvexHull fails and would return nan, returning 0. Npoints = " << points.size() << endl;
        outfile << "ConvexHull fails and would return nan, returning 0. Npoints = " << points.size() << endl;
        warnings6++;
    }
    // Take the absolute value of the area and divide by 2
    // ---------------------------------------------------
    Area = fabs(Area) / 2.0;
    return Area;
}

// Compute utility function, part describing the cost of available area
// NB: we cannot extend the array indefinitely in R, so we encode this into a price to pay for the total
// area that the detectors occupy. We compute the area of the convex hull, which is easily differentiable
// with respect to the x,y of the detectors. 
// ------------------------------------------------------------------------------------------------------
void ComputeUtilityArea (int id=0, int mode=0) {
    double AreaDampCoeff = pi*1.44E6; // 1.2 km radius array 
    double HullArea; 
    if (mode==0) { // Compute utility
        // Get hull area
        // -------------
        HullArea = ConvexHull();
        // We pay no price for an area smaller than areadampcoeff, and an increasingly larger price thereafter
        // if (HA<=ADC) U_TA = 0.
        // if (HA>ADC)  U_TA = -coeff_TA * (exp((HA-ADC)/ADC)-1)
        // (save the coeff_TA): dU_TA/dx = -1/ADC * exp((HA-ADC)/ADC) * dHA/dx
        // ---------------------------------------------------------------------------------------------------
        if (HullArea<=AreaDampCoeff) { 
            U_TA = 0.;
        } else {
            U_TA = -coeff_TA*(exp((HullArea-AreaDampCoeff)/AreaDampCoeff)-1.);
        }
        cout << endl;
        outfile << endl;
        cout    << "     Hull area ratio = " << HullArea/AreaDampCoeff << " U_TA = " << U_TA;
        outfile << "     Hull area ratio = " << HullArea/AreaDampCoeff << " U_TA = " << U_TA;
    } else if (mode==1) { // Derivative wrt x[id] - note, Coeff_TA is not a factor in the derivative
        // Get hull area after dx move of detector id
        // ------------------------------------------
        HullArea = ConvexHull();
        double dx = maxDispl/50.; // Just a number - must not be large
        // Note, we cannot just compute the area var for positive dx, as we could be in a situation where two dets
        // occupy the same point, or when the detector is collinear with two others on the envelope and between them:
        // in such cases the derivative could be zero for positive dx, and nonzero for negative dx, so we need to check both cases.
        // ------------------------------------------------------------------------------------------------------------------------
        double AreaPludx = ConvexHull (id,dx,0);
        double AreaMindx = ConvexHull (id,-dx,0); 
        double dAdx = (AreaPludx-AreaMindx)/(2.*dx); 
        // The above definition is correct and it accounts for cases when both a positive and a negative dx would increase the
        // hull area (e.g. when two detectors are both at one vertex of the hull in a triangular configuration such that moving one in each
        // +dx or -dx positions would cause increases of the area, as the other stays at the original place). 
        // -------------------------------------------------------------------------------------------------------------------------------- 
        if (AreaPludx<=AreaDampCoeff && AreaMindx<=AreaDampCoeff) {
            dUA_dx = 0.;
        } else {
            dUA_dx = -1./AreaDampCoeff*exp((HullArea-AreaDampCoeff)/AreaDampCoeff)*dAdx; 
            //cout << "Ap, am, dUA_dx, HullArea, dAdx = " << AreaPludx << " " << AreaMindx << " " << dUA_dx << " " << HullArea << " " << dAdx << endl;
        }
    } else if (mode==2) {
        // Get hull area after dy move of detector id
        // ------------------------------------------
        HullArea = ConvexHull();
        double dy = maxDispl/50.; // Small number, otherwise all movements increase area
        double AreaPludy = ConvexHull (id,0,dy);
        double AreaMindy = ConvexHull (id,0,-dy); 
        double dAdy = (AreaPludy-AreaMindy)/(2.*dy);
        if (AreaPludy<=AreaDampCoeff && AreaMindy<=AreaDampCoeff) {
            dUA_dy = 0.;
        } else {
            dUA_dy = -1./AreaDampCoeff*exp((HullArea-AreaDampCoeff)/AreaDampCoeff)*dAdy;
            //cout << "Ap, am, dUA_dy, HullArea, dAdy = " << AreaPludy << " " << AreaMindy << " " << dUA_dy << " " << HullArea << " " << dAdy << endl;
        } 
    }
    return;
}

// Different version of extension-sensitive Utility cost contribution. We compute the linear sum of
// distances of detectors from the center of the array - rationale being cables cost, but also connected
// to extension of the impacted area.
// -----------------------------------------------------------------------------------------------------
void ComputeUtilityLength (int id=0, int mode=0) {
    double LengthDampCoeff = 800.*Nunits; // For a uniform distribution within a radius 1.2km, the expected distance is 2/3 r = 800m
    // Get sum of lengths
    // ------------------
    double TotalLength = 0.;
    for (int i=0; i<Nunits; i++) {
        TotalLength += sqrt(x[i]*x[i]+y[i]*y[i]);
    }
    if (mode==0) {
        if (TotalLength<=LengthDampCoeff) {
            U_TL = 0.;
        } else {
            U_TL = -coeff_TL*(exp((TotalLength-LengthDampCoeff)/LengthDampCoeff)-1.); 
        }
        cout << endl;
        outfile << endl;
        cout    << "     Cable length ratio = " << TotalLength/LengthDampCoeff  << " U_TL = " << U_TL;
        outfile << "     Cable length ratio = " << TotalLength/LengthDampCoeff  << " U_TL = " << U_TL;
    } else if (mode==1) { // Note, Coeff_TL is not a factor in the derivative
        double r = sqrt(x[id]*x[id]+y[id]*y[id]);
        double dx = maxDispl/50.;
        double rplu = sqrt(pow(x[id]+dx,2.)+pow(y[id],2.));
        double rmin = sqrt(pow(x[id]-dx,2.)+pow(y[id],2.));
        if (TotalLength+rplu-r<=LengthDampCoeff && TotalLength+rmin-r<=LengthDampCoeff) {
            dUL_dx = 0.;
        } else {
            dUL_dx = -1./LengthDampCoeff*exp((TotalLength-LengthDampCoeff)/LengthDampCoeff)*x[id]/TotalLength; // dLdx = x[id]/TotalLength
        }
    } else if (mode==2) {
        double r = sqrt(x[id]*x[id]+y[id]*y[id]);
        double dy = maxDispl/50.;
        double rplu = sqrt(pow(x[id],2.)+pow(y[id]+dy,2.));
        double rmin = sqrt(pow(x[id],2.)+pow(y[id]-dy,2.));
        if (TotalLength+rplu-r<=LengthDampCoeff && TotalLength+rmin-r<=LengthDampCoeff) {
            dUL_dy = 0.;
        } else {
            dUL_dy = -1./LengthDampCoeff*exp((TotalLength-LengthDampCoeff)/LengthDampCoeff)*y[id]/TotalLength; // dLdy = y[id]/TotalLength
        }
    }
    return;
}

// Compute utility function, gamma fraction part
// ---------------------------------------------
void ComputeUtilityGF () {
    U_GF = coeff_GF * MeasFg / MeasFgErr * ExposureFactor; // Scale with inverse sqrt(density of showers)
    cout    << " U_GF = " << U_GF;
    outfile << " U_GF = " << U_GF;
    return;
}

// Compute utility function and gradient, integrated resolution version
// --------------------------------------------------------------------
void ComputeUtilityIR () {
    U_IR        = 0.;
    U_IR_Num    = 0.;
    U_IR_Den    = 0.;
    for (int is=Nevents; is<Nevents+Nbatch; is++) {
        if (IsGamma[is] && Active[is]) {
            // We compute the U_IR piece considering that each batch event contributes to the estimate of the
            // integrated resolution with a weight (1+wsl*log())*PActive, where PActive accounts for the 
            // probability of that event to have made it to the set. When differentiating wrt xi, we then
            // include the contribution of dPActive/dx,dy
            // ----------------------------------------------------------------------------------------------
            double weight = PActive[is] * (1.+Wslope*log(TrueE[is]/Emin)); 
            U_IR_Den += weight /(InvRmsE[is] * TrueE[is]); // Sum of w*sigma/E
            U_IR_Num += weight;
        }
    }
    if (U_IR_Den!=0.) U_IR = coeff_IR * U_IR_Num / U_IR_Den; 
    cout    << " U_IR = " << U_IR; 
    outfile << " U_IR = " << U_IR;
    return;
}

// The routine below is experimental, not to be used for now
// ---------------------------------------------------------
void ComputeUtilityIR_new () {
    U_IR        = 0.;
    U_IR_Num    = 0.;
    U_IR_Den    = 0.;
    for (int is=Nevents; is<Nevents+Nbatch; is++) {
        if (IsGamma[is] && Active[is]) {
            // We compute the U_IR piece considering that each batch event contributes to the estimate of the
            // integrated resolution with a weight (1+wsl*log())*PActive, where PActive accounts for the 
            // probability of that event to have made it to the set. When differentiating wrt xi, we then
            // include the contribution of dPActive/dx,dy
            // ----------------------------------------------------------------------------------------------
            double weight = PActive[is] * (1.+Wslope*log(TrueE[is]/Emin)); 
            double de_e = fabs(TrueE[is]-e_meas[is][0])/TrueE[is];
            // Winsorize to avoid outliers from having too much weight
            // -------------------------------------------------------
            if (de_e>0.5) de_e = 0.5;
            U_IR_Den += weight * de_e; // Sum of w*(sigma/E)
            U_IR_Num += weight;
        }
    }
    if (U_IR_Den!=0.) U_IR = coeff_IR * U_IR_Num / U_IR_Den; 
    cout    << " U_IR = " << U_IR; 
    outfile << " U_IR = " << U_IR;
    return;
}

// Compute utility function and gradient, pointing resolution version
// ------------------------------------------------------------------
void ComputeUtilityPR () {
    U_PR        = 0.;
    U_PR_Num    = 0.;
    U_PR_Den    = 0.;
    for (int is=Nevents; is<Nevents+Nbatch; is++) {
        if (IsGamma[is] && Active[is]) {
            // We compute the U_PR piece considering that each batch event contributes to the estimate of the
            // pointing resolution with a weight (1+wsl*log())*PActive, where PActive accounts for the 
            // probability of that event to have made it to the set. When differentiating wrt xi, we then
            // include the contribution of dPActive/dx,dy
            // ----------------------------------------------------------------------------------------------
            double weight = PActive[is] * (1.+Wslope*log(TrueE[is]/Emin)); 
            double dp = pi-fabs(fabs(TruePhi[is]-phmeas[is][0])-pi);
            double dr = sqrt(pow(TrueTheta[is]-thmeas[is][0],2.) + pow(sin(TrueTheta[is])*dp,2.)+deltapr2); 
            U_PR_Num += weight * deltapr / dr;
            U_PR_Den += weight;
            // cout << is << " " << PActive[is] << " " << dr << " " << U_PR_Num << endl;
        }
    }
    if (U_PR_Den!=0.) U_PR = coeff_PR * U_PR_Num / U_PR_Den; 
    cout    << " U_PR = " << U_PR;
    outfile << " U_PR = " << U_PR;
    return;
}

// Use N(3s) from model of poisson tail prob with gaussian uncertainty marginalization, see scan_b.C program 
// The model takes B and s_B as mean and uncertainty of mean of a Poisson, and determines N(3s) from a quadratic fit:
// N(3s) = p0(B) + p1(B)*s_B+p2(B)*s_b^2
// Parameters P0, P1 and P2 are modeled with 3-par power curves:
// P0(B) = p00 + p01*pow(B)^p02
// P1(B) = p10 + p11*pow(B)^p12
// P2(B) = p20 + p21*pow(B)^p22
// ------------------------------------------------------------------------------------------------------------------
double N_3S (int id=0, int mode=0) { // WARNING - only call after PS_ variables have been filled correctly

    // If we are in Gaussian regime, use that
    // --------------------------------------
    if (PS_sigmaBoverB>0.3) {
        cout    << "Warning, background uncertainty is too large" << endl;
        outfile << "Warning, background uncertainty is too large" << endl;   
        warnings7++;
    }
    if (PS_B>50. || PS_sigmaBoverB>0.3) {
        double n3s = 1.;
        double sbapprox = 0.;
        // Handle problematic cases
        // ------------------------
        if (PS_B*PS_sigmaBoverB>0.) {
            sbapprox = 3.*sqrt(PS_B+pow(PS_sigmaBoverB*PS_B,2.));
            n3s = PS_B + sbapprox;
        } else {
            cout    << "Warning, PS_B*PS_sigmaBoverB is negative." << endl;
            outfile << "Warning, PS_B*PS_sigmaBoverB is negative." << endl;
            warnings7++;
            if (mode==0) {
                return n3s;
            } else {
                return 0.;
            }
        }
        if (mode==0) {
            return n3s;
        } else if (mode==1) {
            double dn3s_dx = PS_dBdx + 3./(2.*sbapprox)*(PS_dBdx+2.*(PS_sigmaBoverB*PS_B)*(PS_dBdx*PS_sigmaBoverB+PS_B*PS_dsigmaBoverB_dx)); 
            return dn3s_dx;
        } else if (mode==2) {
            double dn3s_dy = PS_dBdy + 3./(2.*sbapprox)*(PS_dBdy+2.*(PS_sigmaBoverB*PS_B)*(PS_dBdy*PS_sigmaBoverB+PS_B*PS_dsigmaBoverB_dy)); 
            return dn3s_dy;
        }
        cout    << "Warning N3S - incorrect parameters: " << PS_sigmaBoverB << " " << PS_B << endl;
        outfile << "Warning N3S - incorrect parameters: " << PS_sigmaBoverB << " " << PS_B << endl;
        warnings6++;
        return 1.;
    } else { // Use interpolated values of lognormal nuisance model  
        double p0 = N3s_p00+N3s_p01*pow(PS_B,N3s_p02);
        double p1 = N3s_p10+N3s_p11*pow(PS_B-N3s_p12,2.);
        double p2 = N3s_p20+N3s_p21*pow(PS_B,N3s_p22);
        double N = p0 + p1*PS_sigmaBoverB + p2*pow(PS_sigmaBoverB,2.);
        if (mode==0) {
            if (N<1.) {
                N = 1; // Just protect from some screwups
                cout << "     warning N_3S N<1?" << endl;
            }
            return N;
        } else if (mode==1) { // Return derivative wrt x of detector id
            // By deriving over dx we have
            // d/dx [n] = d/dx (p0) + d/dx (p1*sb_b) + d/dx (p2*sb_b^2) =
            //          = d/dx (p00+p01*b^p02) + d/dx [(p10+p11*(b-p12)^2)*sb_b] + d/dx [(p20+p21*b^p22)*sb_b^2]
            //          = p01*p02*b^(p02-1)*db/dx + 2*p11*(b-p12)*db/dx*sb_b + (p1)*dsb_b/dx + 
            //            p21*p22*b^(p22-1)*db/dx*sb_b^2 + (p2)*2*sb_b*dsb_b/dx
            // -------------------------------------------------------------------------------------------------
            double dn3s_dx = N3s_p01*N3s_p02*pow(PS_B,N3s_p02-1.)*PS_dBdx + 2.*N3s_p11*(PS_B-N3s_p12)*PS_dBdx*PS_sigmaBoverB +
                             p1*PS_dsigmaBoverB_dx + (N3s_p21*N3s_p22*pow(PS_B,N3s_p22-1.))*PS_dBdx*pow(PS_sigmaBoverB,2.) +
                             p2*2.*PS_sigmaBoverB*PS_dsigmaBoverB_dx;
            if (dn3s_dx!=dn3s_dx) {
                cout << "Warning dn3s PS_B = " << PS_B << "PS_dbdx =" << PS_dBdx << " PS_sbob = " << PS_sigmaBoverB << " PS_dsbobdx=" << PS_dsigmaBoverB_dx << endl; 
                warnings3++;
                TerminateAbnormally();
                return 0.;
                //cout << " dn3sx = " << dn3s_dx << " dbdx = " << PS_dBdx << " dsbbdx = " << PS_dsigmaBoverB_dx << endl;
            }
            return dn3s_dx; 
        } else if (mode==2) { // Return derivative wrt y of detector id
            double dn3s_dy = N3s_p01*N3s_p02*pow(PS_B,N3s_p02-1.)*PS_dBdy + 2.*N3s_p11*(PS_B-N3s_p12)*PS_dBdy*PS_sigmaBoverB +
                             p1*PS_dsigmaBoverB_dy + (N3s_p21*N3s_p22*pow(PS_B,N3s_p22-1.))*PS_dBdy*pow(PS_sigmaBoverB,2.) +
                             p2*2.*PS_sigmaBoverB*PS_dsigmaBoverB_dy;
            if (dn3s_dy!=dn3s_dy) {
                cout << "Warning dn3s PS_B = " << PS_B << "PS_dbdy =" << PS_dBdy << " PS_sbob = " << PS_sigmaBoverB << " PS_dsbobdy=" << PS_dsigmaBoverB_dy << endl; 
                warnings3++;
                TerminateAbnormally();
                return 0.;
                //cout << " dn3sy = " << dn3s_dy << " dbdy = " << PS_dBdy << " dsbbdy = " << PS_dsigmaBoverB_dy <<endl;
            }
            return dn3s_dy; 
        }
        cout << "Warning N3S - incorrect parameters" << endl;
        return 1.;
    }
}

// Service routine to check approximation of 3-sigma counts
// --------------------------------------------------------
void plot_agreement_3s () {
    TH2D * Ratio = new TH2D ("Ratio","",100, 0., 300., 100, -0.1, 20.1);
    for (int i=0; i<100; i++) {
        PS_B = i*3.+1.5;
        for (int j=0; j<101; j++) {
            double sb = j/500.*PS_B;
            PS_sigmaBoverB = sb/PS_B;
            double n_3sigma = PS_B+3.*sqrt(PS_B+sb*sb);
            double N3S = N_3S();
            double R = N3S/n_3sigma;
            Ratio->SetBinContent(i+1,j+1,R);
            cout << " Doing i,j = " << PS_B << " " << PS_sigmaBoverB << " N3s = " << N3S << " G3s = " << n_3sigma << " R = " << R << endl; 
        }
    }
    Ratio->Draw("COLZ");
    return;
}

// Compute utility function and gradient, specified as required integration time to reach 5-sigma counting significance
// for a monochromatic pointlike source located on the celestial equator (we also assume the detector is located at the
// earth equator for simplicity). We use an approximated formula for significance, Z = 2 (sqrt(S+B)-sqrt(B)) and we
// imagine a background estimate obtained from a sideband region 100 times wider than the signal region, (10xdphi*10xdtheta),
// at the same energy of the source. The search region is thus 2.8 sigma wide in E, theta, phi, and the background is thus counted if it has 
// energy = {Esource-1.4 sigmaE, Esource+1.4 sigmaE), phi = (phisource-5*1.4sigmaphi,phisource+5*1.4sigmaphi), and
// theta = (thetasource-5*1.4sigmatheta, thetasource+5*1.4sigmatheta). By inverting the significance formula and inserting Z=5
// we get N = 0.25*[4*B+(5+2*sqrt(B))^2].   
// -----------------------------------------------------------------------------------------------------------------------------------------
void ComputeUtilityPeVSource () {

    // We compute the expected background in the signal region by using a 3D window in E, theta, phi, sized (1,10,10) times the signal region itself,
    // which extends by +- 1.4 sigma in each of the six directions. The sigma_theta, sigma_phi are all-sky estimates, as the source moves in the sky.
    // Instead the sigma_E estimate is computed using only photons with true energy within 50% of the energy of the point source.
    // (here we are looping on batch events to get those sigmas, but we should do it on the training data - to be fixed (it needs fixing also
    // elswhere where we calculate derivatives))
    // ----------------------------------------------------------------------------------------------------------------------------------------------
    U_PS             = 0.;
    double avedth    = 0.;
    double avedph    = 0.;
    double sumde     = 0.;
    double vardth    = 0.;
    double vardph    = 0.;
    double sumde2    = 0.;
    double sumweight = 0.;
    double sumweighte= 0.;
    double weight, tmpdt, tmpdp, tmpde;
    // Variables PS_* are declared static at the start of the code, and get defined here. They are used in the derivative calculation
    // ------------------------------------------------------------------------------------------------------------------------------
    for (int id=0; id<Nunits; id++) {
        PS_dnum1dxj[id]    = 0.;
        PS_dnum2dxj[id]    = 0.;
        PS_dnum1dyj[id]    = 0.;
        PS_dnum2dyj[id]    = 0.;
        PS_sumdPAkdxj[id]  = 0.;
        PS_sumdPAkdyj[id]  = 0.;
        PS_sumdPAkdxjw[id] = 0.;
        PS_sumdPAkdyjw[id] = 0.;
    }
    // NNBB below we use train data to extract variance estimates, as this is cleaner
    // ------------------------------------------------------------------------------
    for (int is=0; is<Nevents; is++) {
        if (IsGamma[is] && Active[is]) {
            weight = PActive[is];
            tmpdt = TrueTheta[is]-thmeas[is][0];
            tmpdp = pi-fabs(fabs(TruePhi[is]-phmeas[is][0])-pi);
            avedth += weight*tmpdt;
            avedph += weight*tmpdp;
            vardth += weight*tmpdt*tmpdt;
            vardph += weight*tmpdp*tmpdp;
            sumweight += weight;
            // Only consider gammas in vicinity of source energy for variance calculation
            // --------------------------------------------------------------------------
            if (fabs(TrueE[is]-E_PS)<0.5*E_PS) {
                tmpde = TrueE[is]-e_meas[is][0];
                sumde  +=weight*tmpde;         // num2
                sumde2 +=weight*tmpde*tmpde;   // num1
                sumweighte += weight;          // den
                // Compute also arrays that are needed for derivatives calculations later
                // ----------------------------------------------------------------------
                for (int id=0; id<Nunits; id++) {
                    // For the sigma_Ei derivative we write sigma2_Ei = num1/den - (num2/den)^2 from which:
                    //     dsigma2_Ei/dxj = (dnum1_dxj * den - num1*dden_dxj)/den^2 - 2*(num2/den)*(dnum2_dxj*den-num2*dden_dxj)/den^2
                    // with
                    //     num1 = sum_k (PA_k*(Et-Ek)^2) = sumde2
                    //     num2 = sum_k (PA_k*(Et-Ek)) = sumde
                    //     den  = sum_k PA_k = sumweighte
                    //     dnum1/dxj = sum_k [dPA_k/dxj * (Et-Ek)^2 - 2*PA_k*(Et-Ek)*dEk/dxj]
                    //     dnum2/dxj = sum_k [dPA_k/dxj * (Et-Ek) - PA_k*dEk/dxj]
                    //     dden = sum_k dPA_k/dxj
                    // ---------------------------------------------------------------------------------------------------------------
                    double dEdR = dEk_dRik (id, is);
                    double dEk_dxj = dEdR * EffectiveDistance (x[id],y[id],x0meas[is][0],y0meas[is][0],thmeas[is][0],phmeas[is][0],1);
                    double dEk_dyj = dEdR * EffectiveDistance (x[id],y[id],x0meas[is][0],y0meas[is][0],thmeas[is][0],phmeas[is][0],2);
                    double dPAk_dxj = 0.;
                    double dPAk_dyj = 0.;
                    if (fabs(SumProbGe1[is]-Ntrigger)<SumProbRange) { 
                        dPAk_dxj = ProbTrigger (SumProbGe1[is],1,id,is); // derivative wrt dx
                        dPAk_dyj = ProbTrigger (SumProbGe1[is],2,id,is); // derivative wrt dy
                        // cout << "dpadx = " << dPAk_dxj << endl;
                    }
                    PS_sumdPAkdxj[id]  += dPAk_dxj;
                    PS_sumdPAkdyj[id]  += dPAk_dyj;
                    PS_sumdPAkdxjw[id] += dPAk_dxj*weight;
                    PS_sumdPAkdyjw[id] += dPAk_dyj*weight;
                    PS_dnum1dxj[id]    += dPAk_dxj*tmpde*tmpde - 2.*weight*tmpde*dEk_dxj; // dnum1 dx. Minus sign because tmpde = Et-Ek and we derive wrt the latter
                    PS_dnum2dxj[id]    += dPAk_dxj*tmpde* - weight*dEk_dxj;               // dnum2 dx. Same as above for minus sign
                    PS_dnum1dyj[id]    += dPAk_dyj*tmpde*tmpde - 2.*weight*tmpde*dEk_dyj; // dnum1 dy. Ditto
                    PS_dnum2dyj[id]    += dPAk_dyj*tmpde* - weight*dEk_dyj;               // dnum2 dy. Ditto
                }
            }
        } 
    }
    avedth = avedth/sumweight;
    avedph = avedph/sumweight;
    double avede  = sumde/sumweighte;
    PS_vardth = vardth/sumweight - avedth*avedth;
    PS_vardph = vardph/sumweight - avedph*avedph;
    PS_varde  = sumde2/sumweighte - avede*avede;
    // We use the values computed above for the derivative calculation. Note that varde, avede are not divided by sumweighte yet.
    // Also note, we get dsigma_E/dx from dsigma2_E/dx by multiplying by 1/2(sigma_E) because sigma_E = sqrt(sigma2_E)
    // --------------------------------------------------------------------------------------------------------------------------
    for (int id=0; id<Nunits; id++) {
        PS_dsigmaEi_dxj[id] = pow(2.*PS_varde,-0.5)*((PS_dnum1dxj[id]*sumweighte-sumde2*PS_sumdPAkdxj[id])/pow(sumweighte,2.) - 
                              2.*(sumde/sumweighte)*(PS_dnum2dxj[id]*sumweighte-sumde*PS_sumdPAkdxj[id])/pow(sumweighte,2.));     // check this!!
        PS_dsigmaEi_dyj[id] = pow(2.*PS_varde,-0.5)*((PS_dnum1dyj[id]*sumweighte-sumde2*PS_sumdPAkdyj[id])/pow(sumweighte,2.) - 
                              2.*(sumde/sumweighte)*(PS_dnum2dyj[id]*sumweighte-sumde*PS_sumdPAkdyj[id])/pow(sumweighte,2.));     // check this!!
    }

    // Now we have the variance of the energy measurement for photons around the source energy, and
    // we can get the relative flux in that interval. We do it on batch data (as opposed to calc above)
    // ------------------------------------------------------------------------------------------------
    PS_SidebandArea = pow(20.*1.4,2.)*sqrt(PS_vardth*PS_vardph);
    PS_TotalSky = (1.-cos(thetamax))*twopi;
    PS_EintervalWeight = 0.;
    PS_SumWeightsE     = 0.;
    PS_SumWeightsE2    = 0.;
    cout << endl << "     Variances = " << PS_vardth << " " << PS_vardph << " " << PS_varde << endl;
    for (int is=Nevents; is<Nevents+Nbatch; is++) {
        if (IsGamma[is] && Active[is]) {
            weight = PActive[is]; // No wslope term, as we base our calculation on events in the window
            if (fabs(e_meas[is][0]-E_PS)<1.4*sqrt(PS_varde)) {
                PS_EintervalWeight += weight;
            }
            PS_SumWeightsE += weight;
            PS_SumWeightsE2+= weight*weight;    
        }
    }

    // The expected background in the signal region is one hundredth of the flux in the sideband region, for
    // gamma rays of energy within the signal window
    // -----------------------------------------------------------------------------------------------------
    if (PS_SumWeightsE==0.) {
        outfile << "Warning, Sumweights=0 in PS calculation. Terminating." << endl;
        cout    << "Warning, Sumweights=0 in PS calculation. Terminating." << endl;
        warnings3++;
        TerminateAbnormally();
        return;
    }
    // Note: the B estimate is arbitrary (depends on some fixable norm factor determining the flux per year in the simulated energy range).
    // We count the extension of the dphi x dtheta signal region as a fraction of the total sky (0.01*PS_sidebandArea/PS_TotalSky) and the
    // energy fraction of the signal region as another efficiency term PS_EintervalWeight/PS_SumWeightsE, tracking the width in energy.
    // Ultimately the Poisson error we account for is sum_weights^2 for gammas in the batch, assuming they have all generated to fall in
    // the dphi x dtheta sideband (as noted, this is immaterial, as it is modulated anyway by the norm factor). What matters is that the
    // number of photons scales with the evolution of the array and the resolutions, other things being fixed.
    // ------------------------------------------------------------------------------------------------------------------------------------
    PS_B = BackgroundConstant/Nevents*0.01*ExposureFactor*(PS_SidebandArea/PS_TotalSky)*(PS_EintervalWeight/PS_SumWeightsE); // denominator of last factor is the total from which we singled out PeV ones
    cout    << "     B = " << PS_B << " SBA = " << PS_SidebandArea << " Eint/SW = " << PS_EintervalWeight << "/" << PS_SumWeightsE;   
    outfile << "     B = " << PS_B << " SBA = " << PS_SidebandArea << " Eint/SW = " << PS_EintervalWeight << "/" << PS_SumWeightsE;   

    // Now we have an estimate of B given the energy and angular resolutions, and all we need is to use the approximate significance formula
    // solving for 1/N(5s):
    // -------------------------------------------------------------------------------------------------------------------------------------
    if (useN5s) { // use N_5s from Z = 5 = [sqrt(B+N_5s)-sqrt(B)]*2 approximation (no uncert)
        U_PS = coeff_PS * ExposureFactor * 4./(25.+20.*sqrt(PS_B));
    } else { 
        // Use N(3s) from model of poisson tail prob with gaussian uncertainty marginalization, see scan_b.C program 
        // The model takes B and s_B as mean and uncertainty of mean of a Poisson, and determines N(3s) from a quadratic fit:
        // N(3s) = p0(B) + p1(B)*s_B + p2(B)*s_b^2
        // Parameters P0, P1 and P2 are modeled with 3-par power curves:
        // P0(B) = p00 + p01*pow(B)^p02
        // P1(B) = p10 + p11*pow(B)^p12
        // P2(B) = p20 + p21*pow(B)^p22
        // 
        // Below we assume [sigma(N_B)/N_B]^2 = sum(w2)/sum(w)^2 + (sigma_Fs/Fs)^2
        // ----------------------------------------------------------------------- 
        PS_sigmaBoverB = sqrt(PS_SumWeightsE2/(PS_SumWeightsE*PS_SumWeightsE)+pow(MeasFgErr/MeasFg,2.));
        U_PS = coeff_PS * ExposureFactor / N_3S(); 
    }
    cout    << " U_PS = " << U_PS;
    outfile << " U_PS = " << U_PS;
    U_GF = U_PS; // For plotting purposes we store it in the gf piece
    return;
}

// Function to be executed by each thread to generate and reconstruct events
// -------------------------------------------------------------------------
void threadFunction (int threadId) {

    // Avoid single-cpu collapse
    // -------------------------
#if defined(STANDALONE) || defined(UBUNTU)
    SetAffinity(threadId);
#endif

    // In normal running we divide the full set of events into the specified number of threads;
    // However if SameShowers is on, we need to first generate the first Nevents, and then deal
    // with the other half, otherwise when dealing with batch events we cannot address event [is-Nevents] 
    // (to guarantee we get the same x0, y0 for it) while doing event [is], in GenerateShower.
    // Correspondingly, we split event generation into two successive calls to threadFunction in that case.
    // ----------------------------------------------------------------------------------------------------
    int Nperthread = (Nevents+Nbatch)/Nthreads;
    int Remainder  = (Nevents+Nbatch)%Nthreads;
    if (SameShowers && Nthreads>1) {
        Nperthread = Nevents/Nthreads; // If we run with SameShowers, it is set Nevents==Nbatch. See above comment.
        Remainder  = Nevents%Nthreads;
    }
    // Split overflow events evenly
    // ----------------------------
    int ismin, ismax;
    if (threadId<Remainder) {
        ismin = threadId*(Nperthread+1);
        ismax = ismin + (Nperthread+1);        
    } else {
        ismin = Remainder * (Nperthread+1) + (threadId-Remainder) * Nperthread;
        ismax = ismin + Nperthread;
    }
    // Double check that we are doing them all
    // ---------------------------------------
    if (Nthreads>1 && threadId==Nthreads-1) {
        if (SameShowers && (ismax!=Nevents || ismax!=2*Nevents) || (ismax!=Nevents+Nbatch)) {
            cout    << "     Warning, faulty assignment in threadFunction at thread = " << threadId << ", ismax = " << ismax << endl;
            outfile << "     Warning, faulty assignment in threadFunction at thread = " << threadId << ", ismax = " << ismax << endl;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings7++; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
        }
    }
    // Lock the mutex before accessing the shared data
    // std::lock_guard<std::mutex> lock(datamutex);

    // Perform operations on the shared data

    // We compute the templates of test statistic 
    // (log-likelihood ratio) for gammas and protons, and while
    // we are at it, we also generate extra Nbatch events for SGD
    // ----------------------------------------------------------

    for (int is=ismin; is<ismax; is++) {
        if (Nthreads==1 && (is+1)%(Nevents/4)==0) {
            cout    << is+1 << " ";
            outfile << is+1 << " ";
        }

        // Zero LRT and sigmaLRT arrays
        // ----------------------------
        logLRT[is]   = 0.;
        sigmaLRT[is] = 1.;

        // Find Nm[], Ne[] for this event
        // -------------------------------
        double p = myRNG->Uniform();
        if ((is<Nevents && is%2==0) || p<GenGammaFrac) { 
            IsGamma[is] = true;
        } else {
            IsGamma[is] = false;
        }

        // Find x0,y0, and other parameters of this shower
        // -----------------------------------------------
        GenerateShower (is);

        // For this shower, find value of test statistic. NB this requires that we have truex0, truey0,
        // and the mug, eg, mup, ep values filled for this event. GenerateShower takes care of the latter
        // ----------------------------------------------------------------------------------------------
        bool ShowerOK = FindLogLR (is); // Fills logLRT[] array and fills sigmaLRT - NB We rely on mug,eg,mup,ep being def above  
        if (!ShowerOK) {
            Active[is]  = false;
            PActive[is] = 0.;
            return;
        }             
        //cout << " Found LLR " << endl;

        // Luis code (to check initialization of likelihood maximization)
        // -------------------------------------------------------------------------------------------------------
        // define a grid (2,3,5) on the five parameters, AROUND the true value
        // (TrueX0[is], TrueY0[is],....)
        // tryx0, tryy0, tryth, tryph, tryEn
        // IsGamma[is]
        // double L = computelikelihood(tryx0, tryy0,.... isphoton=true)
        // compare with the true maximum (access by calling L_max = computelikelihood(measx0[is][], measy0[],...))
        // compute fraction of times that L_max is actually better than any of the grid points.

    } // End is loop
    if (Nthreads==1) {
        cout    << endl;
        outfile << endl;
    }
}

// Multithreading function computing derivatives of U wrt x,y
// Call with threadId = 0 for no threading 
// ----------------------------------------------------------
void threadFunction2 (int threadId) { 

    // Avoid single-cpu collapse
    // -------------------------
#if defined(STANDALONE) || defined(UBUNTU)
    SetAffinity(threadId);
#endif

    int Nperthread = Nunits/Nthreads;
    int Remainder  = Nunits%Nthreads;

    // Split overflow events evenly
    // ----------------------------
    int idmin, idmax;
    if (threadId<Remainder) {
        idmin = threadId*(Nperthread+1);
        idmax = idmin + (Nperthread+1);        
    } else {
        idmin = Remainder * (Nperthread+1) + (threadId-Remainder) * Nperthread;
        idmax = idmin + Nperthread;
    }
    // Double check that we are doing them all
    // ---------------------------------------
    if (Nthreads>1 && threadId==Nthreads-1) {
        if (idmax!=Nunits) {
            cout    << "     Warning, faulty assignment in threadFunction2 at thread = " << threadId << ", idmax = " << idmax << endl;
            outfile << "     Warning, faulty assignment in threadFunction2 at thread = " << threadId << ", idmax = " << idmax << endl;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings7++; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            idmax = Nunits;
        }
    }

    // Lock the mutex before accessing the shared data
    // std::lock_guard<std::mutex> lock(datamutex);

    // Record start of timing
    // ----------------------
    //std::clock_t starting_time = std::clock(); 
    
    // All calculations take place within the detector loop below
    // ----------------------------------------------------------
    for (int id=idmin; id<idmax; id++) { 

        dU_dxi[id] = 0.;
        dU_dyi[id] = 0.;
        if (keep_fixed[id]) continue;   // Do not move this detector 

        double dldxm[maxEvents];  // This is used to store values during calculations of derivatives, to avoid multiple calculations
        double dldym[maxEvents];  // This is used to store values during calculations of derivatives, to avoid multiple calculations 
        double dpg_dx[maxEvents];
        double dpp_dx[maxEvents];
        double dpg_dy[maxEvents];
        double dpp_dy[maxEvents];
        double dRtot_dx;
        double dRtot_dy;
        double dTotalRspan_dx;
        double dTotalRspan_dy;

        if (Nthreads==1 && (id+1)%(Nunits/10)==0) cout << id+1 << " ";
        double xi  = x[id]; // Speeds up calculations
        double yi  = y[id]; 

        if (scanU && id!=idstar) continue; // Only compute dUdx,dy for that one detector if scanning dU

        // TotalRspan = Rslack + aveR + 2*sqrt(ave(R^2)-aveR*aveR) = Rslack + aveR + 2*rms(R)
        // So when we derive wrt a movement dxi we get:
        // dTotalRspan_dx = d(aveR)/dx + 1/rms(R)*d[var(R)] = 
        //                = d(aveR/dRi*dRi/dxi) + 1/rms(R)*{d[ave(R^2)/dRi*dRi/dxi]-2*aveR*d(aveR/dRi*dRi/dxi)}
        //                = 1/N*xi/Ri + 1/rms(R)*{2Ri/N*xi/Ri - 2*aveR*1/N*xi/Ri}
        //                = 1/N*xi/Ri + 1/rms(R)*{2xi/N - 2*aveR/N xi/Ri}
        // So we get the expression below, where ArrayRspan[1] is aveR, and ArrayRspan[2] is rms(R)
        // ----------------------------------------------------------------------------------------------------
        double Ri = pow(xi,2.)+pow(yi,2.);
        if (Ri>0.) {
            Ri = sqrt(Ri);
            dRtot_dx = xi/(Nunits*Ri); // NB here and below we are concerned with distances on the plane
            dRtot_dy = yi/(Nunits*Ri); // ... and not true det-shower axis distances, as trials are done on the plane
            dTotalRspan_dx = dRtot_dx + 2.*xi/(Nunits*ArrayRspan[2])*(1.-ArrayRspan[1]/Ri);
            dTotalRspan_dy = dRtot_dy + 2.*yi/(Nunits*ArrayRspan[2])*(1.-ArrayRspan[1]/Ri);
        } else {
            dRtot_dx = 1./Nunits; // Derivative of average is 1/N if det is at the origin
            dRtot_dy = 1./Nunits;
            dTotalRspan_dx = dRtot_dx + 2./(Nunits*ArrayRspan[2])*(xi-ArrayRspan[1]);
            dTotalRspan_dy = dRtot_dy + 2./(Nunits*ArrayRspan[2])*(yi-ArrayRspan[1]);
        }

        // First fill the arrays dldxm[][], dldym[][] once and for all
        // -----------------------------------------------------------
        for (int m=0; m<Nevents; m++) {
            if (!Active[m]) continue;
            std::pair<double,double> dlogLRdxy = dlogLR_dxy(id,m);
            dldxm[m] = dlogLRdxy.first;  // dlogLR_dR (id,m,1);
            dldym[m] = dlogLRdxy.second; // dlogLR_dR (id,m,2);
        }

        // New calculation of dpg_dxi, dpg_dyi, dpp_dxi, dpp_dyi, hopefully less messy
        // We have Pg,k = Sum_{m=1..Nevents,g} [PActive_m*G_{den,m}*G(Tm-Tk,s_m)] / Sum_m PActive_m
        // with 
        //      G_{den,m} = (2pi*s_m^2)^{-1/2}
        //      G(Tm-Tk,s_m) = exp(-1/2 (Tm-Tk)^2/s_m^2)
        // We get, by deriving wrt xi:
        //      dPg,k/dxi = { Sum_m [ (PA_m Gden G) * [ (Tm-Tk)/s_m^2 * (dTk/dxi-dTm/dxi) + 
        //                                              (Tm-Tk)^2/(2s_m^4) ds_m^2/dxi +
        //                                              -Gden^2 pi ds_m^2/dxi + 1/PA_m dPA_m/dxi ]] * Sum_m PA_m
        //                    - Sum_m (PA_m Gden G) * Sum_m (dPA_m/dxi) } / (Sum_m PA_m)^2
        // ------------------------------------------------------------------------------------------------------
        for (int k=Nevents; k<Nevents+Nbatch; k++) {
            if (!Active[k]) continue;
            std::pair<double,double> dlogLRdxy = dlogLR_dxy (id,k);
            double dldxk  = dlogLRdxy.first;  // dlogLR_dR (id,k,1);
            double dldyk  = dlogLRdxy.second; // dlogLR_dR (id,k,2);
            double numg     = 0.;
            double deng     = 0.;
            double dnum_dxg = 0.;
            double dnum_dyg = 0.;
            double dden_dxg = 0.;
            double dden_dyg = 0.;
            double nump     = 0.;
            double denp     = 0.;
            double dnum_dxp = 0.;
            double dnum_dyp = 0.;
            double dden_dxp = 0.;
            double dden_dyp = 0.;

            // Get components of dpg/dxi, dpg/dyi
            // ----------------------------------
            for (int m=0; m<Nevents; m++) {
                if (!Active[m]) continue;
                double Sm          = SumProbGe1[m]; 
                double sigma       = sigmaLRT[m];
                double sigma2      = sigma*sigma;
                double sigma4      = sigma2*sigma2;
                double dlmk        = logLRT[m]-logLRT[k];
                double Gden        = 1./(sqrt2pi*sigma); 
                double G           = exp(-pow(dlmk/sigma,2.)/2.); 

                // We recall that Pg is sum_m (P*Gden*G) / sum_m P
                // -----------------------------------------------
                double Factor      = PActive[m]*Gden*G;
                if (Factor!=Factor) {
                    cout    << "Warning, nan in dpdx calculation " << endl;
                    cout    << "Gden, dlmk, sigma, Sm, PA = " << Gden << " " << dlmk << " " << sigma << " " << Sm << " " << PActive[k] << endl;
                    outfile << "Warning, nan in dpdx calculation " << endl;
                    outfile << "Gden, dlmk, sigma, Sm, PA = " << Gden << " " << dlmk << " " << sigma << " " << Sm << " " << PActive[k] << endl;
#if defined(STANDALONE) || defined(UBUNTU)
                    datamutex.lock();
#endif                    
                    warnings4++; 
#if defined(STANDALONE) || defined(UBUNTU)
                    datamutex.unlock();
#endif
                    continue; 
                }

                // Get dPA_dx, dPA_dy
                // To incorporate the contribution of PActive_m, we have made it part of the def of G.
                // This means we have a term to add, G/PActive_m * dPActive_m/dxi. 
                // The latter term can be computed as follows (PActive = 1 - sum(1:Ntr-1)):
                //     dPActive_m/dxi = -d/dxi [sum_{j=0}^{Ntr-1}(e^{-Sm}*Sm^j/j!)]
                // with
                //     Sm = Sum_{i=1}^Ndet [1-exp(-lambda_mu^i-lambda_e^i)_m]
                // We get
                //     dPActive_m/dxi = Sum_{j=0}^{Ntr-1} 1/j! [e^{-Sm}*(Sm^j-j*Sm^(j-1))*dSm/dxi]
                // Now for dSm/dxi we have
                //     dSm/dxi = d/dxi [ Sum_{i=1}^{Ndet} (1-e^(-lambda_mu^i-lambda_e^i)_m))]
                //               = -d/dxi (e^[-lambda_mu^i-lambda_e^i]_m) =
                //               = e^[-lambda_mu^i-lambda_e^i]*(dlambda_mu^i/dxi + dlambda_e^i/dxi)
                // and the latter are computed in the flux routines.
                // Note that the lambdas are the true ones, as we compute PDFs with events that are
                // included in the sum if they pass the trigger, and they do if the true fluxes exceed
                // the threshold of Ntrigger detectors firing.
                // -----------------------------------------------------------------------------------
                // The calculation of the term due to dPActive/dxi is laborious, but we only need
                // to perform it if we are close to threshold, otherwise the derivative contr. is null
                // -----------------------------------------------------------------------------------
                bool consider_dP = false;
                // If we need a number of hit tanks "close" to Ntrigger in order to consider the derivative of PActive over dx, dy,
                // then we may interpret N_hit (of which Sm is an approximation) to be a Poisson variate. This is then close to
                // Ntrigger if fabs(Sm-Ntrigger)<2.*sqrt(Ntrigger). E.g. for Ntrigger=10 we get [4,16], for Ntrigger=30 we get [19,41].

                if (fabs(Sm-Ntrigger)<SumProbRange) consider_dP = true; 
                //if (fabs(Sm-Ntrigger)<0.1*Nunits) consider_dP = true; // was 1. might be fine tuned. 1 prolly good
                double pam = PActive[m];
                if (pam>0.) { 
                    double Gden2 = pow(Gden,2.);
                    double ds2dx = dsigma2_dx[id][m];
                    double ds2dy = dsigma2_dy[id][m];
                    if (IsGamma[m]) {
                        dnum_dxg += (dlmk/sigma2*(dldxk-dldxm[m]) + pow(dlmk,2.) / (2.*sigma4) * ds2dx); 
                        dnum_dxg += pi * Gden2 * ds2dx;
                        dnum_dyg += (dlmk/sigma2*(dldyk-dldym[m]) + pow(dlmk,2.) / (2.*sigma4) * ds2dy); 
                        dnum_dyg += pi * Gden2 * ds2dy;
                        if (consider_dP) {
                            double dPAm_dx = ProbTrigger (Sm,1,id,m);
                            double dPAm_dy = ProbTrigger (Sm,2,id,m);
                            dnum_dxg += dPAm_dx / pam;
                            dden_dxg += dPAm_dx;
                            dnum_dyg += dPAm_dy / pam;
                            dden_dyg += dPAm_dy;
                        }
                        dnum_dxg *= Factor;
                        dnum_dyg *= Factor;
                        numg += Factor;
                        deng += pam;
                    } else {
                        dnum_dxp += (dlmk/sigma2*(dldxk-dldxm[m]) 
                                      + pow(dlmk,2.) / (2.*sigma4) * ds2dx); 
                        dnum_dxp += pi * Gden2 * ds2dx;
                        dnum_dyp += (dlmk/sigma2*(dldyk-dldym[m]) 
                                      + pow(dlmk,2.) / (2.*sigma4) * ds2dy); 
                        dnum_dyp += pi * Gden2 * ds2dy;
                        if (consider_dP) {
                            double dPAm_dx = ProbTrigger (Sm,1,id,m);
                            double dPAm_dy = ProbTrigger (Sm,2,id,m);
                            dnum_dxp += dPAm_dx / pam;
                            dden_dxp += dPAm_dx;
                            dnum_dyp += dPAm_dy / pam;
                            dden_dyp += dPAm_dy;
                        }
                        dnum_dxp *= Factor;
                        dnum_dyp *= Factor;
                        nump += Factor;
                        denp += pam;
                    } // End if gamma
                } // End if pactive
            } // End m loop

            // With the above sums, we can compute the x,y derivatives of PDF values for both hypotheses of event k.
            // -----------------------------------------------------------------------------------------------------
            double den2g = pow(deng,2.);
            double den2p = pow(denp,2.);
            if (den2g>0.) {
                dpg_dx[k] = (dnum_dxg*deng-numg*dden_dxg)/den2g;
                dpg_dy[k] = (dnum_dyg*deng-numg*dden_dyg)/den2g;
            }
            if (den2p>0.) {
                dpp_dx[k] = (dnum_dxp*denp-nump*dden_dxp)/den2p;
                dpp_dy[k] = (dnum_dyp*denp-nump*dden_dyp)/den2p;
            }
        } // End k loop

        // Now get variation of inverse sigma_fs over dR
        // ---------------------------------------------
        double d_invsigfs_dx  = 0.;
        double d_invsigfs_dy  = 0.;
        double d_fs_dx        = 0.;
        double d_fs_dy        = 0.;

        // Also deal with variation of integrated resolution on measured gamma energy (IR) and on measured angles (PR)
        // -----------------------------------------------------------------------------------------------------------
        double sumdPAkdxi          = 0.;
        double sumdPAkdyi          = 0.;
        double sumdPAkdxi_noweight = 0.;
        double sumdPAkdyi_noweight = 0.;
        double sumdPAkdxidE2       = 0.;
        double sumdPAkdyidE2       = 0.;
        double sum_dedx            = 0.;
        double sum_dedy            = 0.;
        double sum_dnumPRdx        = 0.;
        double sum_dnumPRdy        = 0.;
        double PS_sumdPAkdxi       = 0.;
        double PS_sumdPAkdyi       = 0.;
        double PS_sumdPAkdxiw      = 0.;
        double PS_sumdPAkdyiw      = 0.;
        double dR                  = 0.5*DetectorSpacing; // Use a relevant distance for incremental ratio

        // The stuff below is needed to compute the derivative of U_PS
        // For U_PS we proceed as follows:
        //   U_PS = 4 / (25+20sqrt(B)) = 1/N(5sigma)
        // The above is an approximation coming from Z = 2(sqrt(S+B)-sqrt(B)) from which, for Z = 5,
        //   5 = 2 sqrt (N_5s + B) - sqrt(B).
        // We also assume we take a 100-times-wider sky region to extrapolate B from, where a flux fraction 
        // of A_sb/A_tot is recorded in a certain time interval. The actual background in the signal region
        // is a fraction E_fr of the total weighted sum of PA(i) for photons:
        //   E_fr = sum_i,g PA_i * Int(Emin,Emax) P(Ei) dE, so 
        //   B = 0.01 A_sb/A_tot * sum_i,g (PA_i Int(Emin,Emax) {P(Ei) dE} / sum_i,g (PA_i)
        // with P(Ei) a gaussian kernel for true energy E (and fixed sigma_E):
        //   P(Ei) = 1/sqrt(2*pi*s_E^2)*exp[-(E-Ei)^2/(2s_E^2)]
        // and Emin = E-1.4 sigma_E, Emax = E+1.4 sigma_E.
        // (NNBB these Emin, Emax are not the statics defining the full range of generated showers)
        // We thus have:
        //   dU_PS/dxj = dU_PS/dB * B * (1/A_sb dA_sb/dxj + 1/E_fr * dE_fr/dxi - 1/sum_i,g(PA_i) dsum_i,g PA_i/dxj )           
        // with:
        //   dU_PS/dB = -40/sqrt(B)/(25+20sqrt(B))^2 
        // And since we define the signal region to be 1.4 sigma wide in theta and phi, we get:
        //   A_sb = (20*1.4)^2 sigma_theta sigma_phi
        // while the total flux on the array is
        //   A_tot = 2*pi*(1-cos(thetamax))  
        // ---------------------------------------------------------------------------------------------------------
        double sum_dnum1s2t_dx = 0.;
        double sum_dnum2s2t_dx = 0.;
        double sum_dden1s2t_dx = 0.;
        double sum_dden2s2t_dx = 0.;
        double sum_dnum1s2t_dy = 0.;
        double sum_dnum2s2t_dy = 0.;
        double sum_dden1s2t_dy = 0.;
        double sum_dden2s2t_dy = 0.;
        double sum_dnum1s2p_dx = 0.;
        double sum_dnum2s2p_dx = 0.;
        double sum_dden1s2p_dx = 0.;
        double sum_dden2s2p_dx = 0.;
        double sum_dnum1s2p_dy = 0.;
        double sum_dnum2s2p_dy = 0.;
        double sum_dden1s2p_dy = 0.;
        double sum_dden2s2p_dy = 0.;
        double Int_Ei          = 0.;
        double dEfrdxj_factor1 = 0.;
        double dEfrdyj_factor1 = 0.;
        double dEfrdxj_factor2 = 0.;
        double dEfrdyj_factor2 = 0.;        
        double sum_num1s2t     = 0.;
        double sum_num1s2p     = 0.;
        double sum_num2s2t     = 0.;
        double sum_num2s2p     = 0.;
        double sum_dens2       = 0.;

        for (int k=Nevents; k<Nevents+Nbatch; k++) {
            
            // Deal with contribution to dUdx due to variation of density. This does not
            // require showers to be active, as we need to consider what changes in pactive
            // for all of them. Compare to the U_IR derivative calculation below, see note.
            // When we need to assess how
            // showers contribute to the U_GF piece (there is a density factor 1/sqrt(Nactive) there),
            // we need to compute dpactive/dxi and there we want to sum on ALL batch showers, as that
            // number is accessible also for showers we did not reconstruct.
            // ----------------------------------------------------------------------------
            double dPAk_dx = 0.;
            double dPAk_dy = 0.;
            if (fabs(SumProbGe1[k]-Ntrigger)<SumProbRange) {   
                dPAk_dx = ProbTrigger (SumProbGe1[k],1,id,k); // Derivative wrt dx
                dPAk_dy = ProbTrigger (SumProbGe1[k],2,id,k); // Derivative wrt dy
            }
            //cout << "dpak_dx = " << dPAk_dx << " " << " dpak_dy = " << dPAk_dy << " SumP, id, k = " << SumProbGe1[k] << " " << id << " " << k << endl;
            sumdPAkdxi_noweight += dPAk_dx; 
            sumdPAkdyi_noweight += dPAk_dy; 

            // All the other calculations are done only for active showers
            // -----------------------------------------------------------
            if (!Active[k]) continue;

            double sqrden = MeasFg * pg[k] + (1.-MeasFg)*pp[k]; 
            if (sqrden<epsilon) continue; // Protect against PDF holes
            double den    = pow(sqrden,2.);
            double dif    = pg[k]-pp[k];
            double dif2   = pow(dif,2.);
            double this_dsfs_dx, this_dsfs_dy;
            if (dif!=0. && den!=0.) { // Protect against adding nan to dudx
                // The contributions to d(1/sigmafs)/dx are computed by remembering that we found already
                // 1/sigmafs = sum_k [(pg_k-pp_k)^2/(MeasFg*pg_k+(1-MeasFg)*pp_k)^2]
                // --------------------------------------------------------------------------------------
                this_dsfs_dx = (dif*(dpg_dx[k]-dpp_dx[k])*den - dif2*sqrden*(MeasFg*dpg_dx[k]+(1.-MeasFg)*dpp_dx[k])) / pow(den,2.);
                this_dsfs_dy = (dif*(dpg_dy[k]-dpp_dy[k])*den - dif2*sqrden*(MeasFg*dpg_dy[k]+(1.-MeasFg)*dpp_dy[k])) / pow(den,2.);
                d_invsigfs_dx += this_dsfs_dx; 
                d_invsigfs_dy += this_dsfs_dy; 
                //if (fabs(this_dsfs_dx)>1.E5 | fabs(this_dsfs_dy)>1.E5) {
                //    cout << "den, dif, dpgdx, dppdx, dppdx, dppdy = " << den << " " << dif << " " << dpg_dx[k] << " " << dpg_dy[k] << " "  << dpp_dx[k] << " " << dpp_dy[k] << endl;  
                //}

                // Now compute dfs_dx, dy. We do it by using implicit differentiation.
                // We treat Fs as the dependent variable and Ps_i, Pp_i independent variables,
                // and use the chain rule to relate derivation by Ps_i, Pp_i to derivation by xi.
                // We define
                //     H = d logL / d Fs = Sum_i [(Ps_i-Pp_i)/(Fs*Ps_i+(1-Fs)*Pp_i)] = 0
                // as the equation defining Fs. We apply derivation by xk, obtaining:
                //     sum_i {dH/dPs_i dPs_i/dxk + dH/dPp_i dPp_i/dxk + dH/dFs dFs/dxk } = 0
                // whence
                //     dFs/dxk = -1/(dH/dFs) * sum_i { dH/dPs_i dPs_i/dxk + dH/dPp_i dPp_i/dxk }
                // Now we have
                //     dH/dFs = - Sum_j [(dif_j)^2/(den_j)^2] = -1/sigmafs^2
                // where we defined dif_j = Ps_j-Pp_j   and den_j = Fs Ps_j + (1-Fs)*Pp_j;
                //     dH/dPs_i = (den_i - dif_i*Fs)/den_i^2 = Pp_i/den_i^2
                //     dH/dPp_i = -(den_i + dif_i*(1-Fs))/den_i^2 = -Ps_i/den_i^2
                // And so
                //     dFs/dxk = { Pp_i * dPs_i/dxk - Ps_i * dPp_i/dxk } / 
                //                {(den_i)^2 * Sum_j [(dif_j)^2/(den_j)^2]}
                // Finally, note that 1./Sum_j [dif^2/den^2] = sigmafs2.
                // ------------------------------------------------------------------------------------------
                d_fs_dx += (pp[k]*dpg_dx[k] - pg[k]*dpp_dx[k])/den; // we further divide by sigma^2 Nbatch later
                d_fs_dy += (pp[k]*dpg_dy[k] - pg[k]*dpp_dy[k])/den;
                // cout << k << " " << d_invsigmafs_dx << " " << d_invsigmafs_dy << " " << dfs_dx << " " << dfs_dy << endl;
            }

            // Now perform calculations that are needed if any of the two resolution pieces are on (see below)
            // -----------------------------------------------------------------------------------------------
            if (IsGamma[k]) {
                double dRdx = 0.;
                double dRdy = 0.;
                double Wk   = 0.;
                double Ek   = e_meas[k][0];
                double x0m  = x0meas[k][0];
                double y0m  = y0meas[k][0];
                double thm  = thmeas[k][0];
                double phm  = phmeas[k][0];
                double tht  = TrueTheta[k];
                double pht  = TruePhi[k];
                if (eta_IR!=0 || eta_PR!=0 || PeVSource) {
                    Wk = 1. + Wslope * log(TrueE[k]/Emin);
                    // if (fabs(SumProbGe1[k]-Ntrigger)<0.1*Nunits) {
                    if (fabs(SumProbGe1[k]-Ntrigger)<SumProbRange) { // was <1. We only consider the derivative of PAk if it has a chance to matter
                        sumdPAkdxi += dPAk_dx * Wk;
                        sumdPAkdyi += dPAk_dy * Wk;
                    }
                    dRdx = -EffectiveDistance (xi,yi,x0m,y0m,thm,phm,1); // nb measured R! And negative sign as it is wrt xi, yi
                    dRdy = -EffectiveDistance (xi,yi,x0m,y0m,thm,phm,2);
                }

                // For PeV Source utility part, we only need to sum these when we are within 50% of the PS energy
                // ----------------------------------------------------------------------------------------------
                if (PeVSource) {
                    // We sum variations of the integral in the signal region by considering all showers that have
                    // even a small chance of contributing if dx, dy bring them closer to E_PS
                    // -------------------------------------------------------------------------------------------

                    // We need to compute the integral Int_Ei = Int_emin^emax P(Ei) dE, with 
                    //     Emin = E_PS-1.4 sigma_E (temporary name, not the min of the energy range for gen!)
                    //     Emax = E_PS+1.4 sigma_E (same, not max of energy range!)
                    //     P(Ei) = exp(-0.5*(Ei-E)^2/sigma_E^2)
                    // We change variables: 
                    //     u = sqrt(0.5)(Ei-E)/s
                    //     du = -sqrt(0.5)/s dE 
                    //     dE = -sqrt(2)*s_E du 
                    //     umin = sqrt(0.5)/s_E*(Ei-E_PS+1.4s_E) 
                    //     umax = sqrt(0.5)/s_E*(Ei-E_PS-1.4s_E) 
                    // and we get the result below as int_a^b(exp(-x^2)dx) = sqrt(pi/2)*(Erf(b)-Erf(a))
                    // (note, the minus sign is reabsorbed in swapping umin and umax, so we get the smaller (umax) to be the lower extremum)
                    // ---------------------------------------------------------------------------------------------------------------------
                    double sigma_E = sqrt(PS_varde); // This is calculated in ComputeUtilityPeVSource, looping on train events
                    double Int_Ei = sqrtpi*sigma_E*(TMath::Erf(Ek-E_PS+1.4*sigma_E)-TMath::Erf(Ek-E_PS-1.4*sigma_E));
                    // For the variation of the fraction of showers in the signal window, dEfr/dx ,dEfr/dy parts,
                    // we need the kernel of Emeasured=Ei evaluated within the E_PS window. The calculation runs as follows:
                    //     Efr = Sum_i,g {PA_i /[sqrt(2*pi)*sigma_E] * Int_emin^emax P(Ei) dE}
                    // with
                    //     Emin = E_PS-1.4 sigma_E (not the min of energy range!)
                    //     Emax = E_PS+1.4 sigma_E (not the max of energy range!)
                    //     P(Ei) = exp(-0.5*(Ei-E)^2/sigma_E^2)
                    //     sigma_E = sqrt { sum_k (PA_k*(Et-Ek)^2) / sum_k PA_k - [sum_k (PA_k*(Et-Ek)) / sum_k PA_k]^2 }
                    // where we note that we put outside the integral the normalization factor with sigma_E which does not vary for 
                    // each showers' kernel (i.e. in reality P(Ei) would include that norm factor but we keep it out of the integral).
                    // NNBB Note that in the sums of sigma_E we take only showers k that are to within 50% of E_PS. The sigma_E that
                    // results is still a function of xj, but we need to sum only within that window in computing the various terms.
                    //
                    // We derive wrt xj:
                    //     dEfr/dxj = sum_i,g { [dPA_i/dxj / [sqrt2pi*sigma_E] - PA_i / (sqrt2pi*sigma_E^2) * dsigma_E/dxj]* I(Ei) + 
                    //                          PA_i/[sqrt2pi*sigma_E] * dI(Ei)/dxj } = sum_i,g (dEfrdx_factor1 + dEfrdx_factor2)  
                    // where since the showers within the signal region are those less than 1.4 sigma_E away from E_PS, and we got PS_EintervalWeight
                    // by selecting those in the batch, we can obtain these sums by considering showers passing a much more loose req. (say 5*sigma_E).
                    // in the batch.
                    // We now need to compute the derivative of the integral with respect to xj and yj. 
                    // We can write, using Leibnitz' rule:
                    // (1) dI(Ei)/dxj = Int_Emin^Emax dP(Ei)/dxj dE + P(Emax) dEmax/dxj - P(Emin) dEmin/dxj
                    // Now we have the following pieces to take care of:
                    // (2) dP(Ei)/dxj = exp(-0.5*(Ei-E)^2/sigma_E^2) * [dEi/dxj * (-(Ei-E)/sigma_E^2) + dsigma_E/dxj*((Ei-E)^2*sigma_E^(-3)) ]
                    //     P(Emax) = exp(-0.5*(Ei-Emax)^2/sigma_E^2)
                    //     P(Emin) = exp(-0.5*(Ei-Emin)^2/sigma_E^2)
                    //     dEmax/dxj = 1.4*dsigma_E/dxj
                    //     dEmin/dxj = -1.4*dsigma_E/dxj 
                    //
                    // For the sigma_E derivative we write sigma_E = num1/den - (num2/den)^2 from which:
                    //     dsigma_E/dxj = (dnum1/dxj * den - num1*dden/dxj)/den^2 - 2*(num2/den)*(dnum2/dxj*den-num2*dden/dxj)/den^2
                    // with
                    //     num1 = sum_k (PA_k*(Et-Ek)^2)
                    //     num2 = sum_k (PA_k*(Et-Ek))
                    //     den  = sum_k PA_k
                    //     dnum1/dxj = sum_k [dPA_k/dxj * (Et-Ek)^2 - 2*PA_k*(Et-Ek)*dEk/dxj]
                    //     dnum2/dxj = sum_k [dPA_k/dxj * (Et-Ek) - PA_k*dEk/dxj]
                    //     dden = sum_k dPA_k/dxj
                    //
                    // There remains to compute the integral from Emin to Emax of dP(Ei)/dxj. We need two pieces from eq.(1,2):
                    //     1) Integral_Emin^Emax dE [ - exp(-0.5*(Ei-E)^2/sigma_E^2) ] * (Ei-E) * dEi/dxj / sigma_E^2
                    // This has an easier primitive, so we proceed to compute it:
                    //     = [- dEi/dxj * exp(-0.5*(Ei-E)^2/sigma_Ei^2)]_Emin^Emax
                    //     2) Integral_Emin^Emax dE [exp(-0.5*(Ei-E)^2/sigma_E^2) ] * (Ei-E)^2 / sigma_E^3 * {dsigma_E/dxj}
                    // This can be solved using the erf, knowing that: 
                    //     Integral x^2 exp(-x^2) dx = [sqrt(pi)/4 Erf(x) -x/2 exp(-x^2)]_xmin^xmax
                    // In our case we have to change variables: 
                    //     u = sqrt(0.5)*[(Ei-E)/s_E]
                    //     du = -sqrt(0.5)/s_E*dE  -->  dE = -sqrt(2)*s_e du
                    // The extrema also get changed: from Emin = E_PS-1.4*s_E, Emax = E_PS+1.4*s_E we get new extrema as
                    //     umin = sqrt(0.5)/s_E*(Ei-E_PS+1.4s_E) 
                    //     umax = sqrt(0.5)/s_E*(Ei-E_PS-1.4s_E) 
                    // The result of the substitution is:
                    //     Integral_(umin)^(umax) (-s_E*sqrt(2) du) [exp(-u^2)]*(2 s_E^2 u^2) / s_E^3 * ds_E/dxj = 
                    //     -Integral_(umin)^(umax) du * [2*sqrt(2) u^2 exp(-u^2)] * ds_E/dxj =
                    //     -2*sqrt(2)* ds_E/dxj * Integral_umin^umax u^2 exp(-u^2) du
                    //     
                    // Now the integral equates to:
                    //     Integral_umin^umax u^2 exp(-u^2) du = 0.5 * int(exp(-x^2)dx) - 0.5 * [x exp(-x^2)]_umin^umax 
                    // So we finally have
                    //     - sqrt(2) * ds_E/dxj * [ sqrt(pi)*Erf(umax) - sqrt(pi)*Erf(umin) - umax * exp(-umax^2) + umin * exp(-umin^2) ]
                    //
                    // Note that in the calculation of the pieces of dEfr_dxi, dyi, we need to have dsigmaE_dxj, dsigmaE_dyj already computed
                    // in order to sum the elements. While sigma_E is fixed (it is computed using showers with +-50% energy from the point source)
                    // its derivative wrt xj, yj has to be obtained on the fly.
                    // ---------------------------------------------------------------------------------------------------------------------------------
                    double E_min = E_PS-1.4*sigma_E;
                    double E_max = E_PS+1.4*sigma_E;
                    double umin = (Ek-E_PS+1.4*sigma_E)/(sqrt2*sigma_E);
                    double umax = (Ek-E_PS-1.4*sigma_E)/(sqrt2*sigma_E); // note swapped - sign due to substitution rule
                    double factor = (sqrt2pi*(TMath::Erf(umax)-TMath::Erf(umin))-sqrt2*(umax*exp(-umax*umax)-umin*exp(-umin*umin)))
                                    + exp(-0.5*pow((Ek-E_max)/sigma_E,2.))*1.4
                                    + exp(-0.5*pow((Ek-E_min)/sigma_E,2.))*1.4;
                    double dInt_Ei_dxj = PS_dsigmaEi_dxj[id] * factor;
                    double dInt_Ei_dyj = PS_dsigmaEi_dyj[id] * factor;
                    //     dEfr/dxj = sum_i,g { [dPA_i/dxj / [sqrt2pi*sigma_E] - PA_i / (sqrt2pi*sigma_E^2) * dsigma_E/dxj]* I(Ei) + 
                    //                          PA_i/[sqrt2pi*sigma_E] * dI(Ei)/dxj } = sum_i,g (dEfrdx_factor1 + dEfrdx_factor2)  

                    // We look for events whose kernel extends into the signal region, to compute how Efr varies with dxj, dyj
                    // -------------------------------------------------------------------------------------------------------
                    if (fabs(Ek-E_PS)<5.*sigma_E) {
                        dEfrdxj_factor1 += (dPAk_dx /(sqrt2pi*sigma_E) - PActive[k]/(sqrt2pi*PS_varde)*PS_dsigmaEi_dxj[id]) * Int_Ei;
                        dEfrdyj_factor1 += (dPAk_dy /(sqrt2pi*sigma_E) - PActive[k]/(sqrt2pi*PS_varde)*PS_dsigmaEi_dyj[id]) * Int_Ei;
                        dEfrdxj_factor2 += PActive[k]/(sqrt2pi*sigma_E) * dInt_Ei_dxj; 
                        dEfrdyj_factor2 += PActive[k]/(sqrt2pi*sigma_E) * dInt_Ei_dyj;
                        //if (dEfrdxj_factor1!=dEfrdxj_factor1) cout << "sigmae, varde, dsigmaei_dx[" << id << "], intEi = " << sigma_E << " " << PS_varde << " " << PS_dsigmaEi_dxj[id] << " " << Int_Ei << endl;
                        //if (dEfrdyj_factor2!=dEfrdyj_factor2) cout << "sigmae, varde, dsigmaei_dy[" << id << "], intEi = " << sigma_E << " " << PS_varde << " " << PS_dsigmaEi_dyj[id] << " " << Int_Ei << endl;
                    }
                    //if (fabs(Ek-E_PS)<1.4*sigma_E) { // As these are needed for the derivative of PS_sumweightsE, which is computed over all the spectrum, we do not cut!
                        PS_sumdPAkdxi += dPAk_dx;
                        PS_sumdPAkdyi += dPAk_dy;
                        PS_sumdPAkdxiw += PActive[k]*dPAk_dx;
                        PS_sumdPAkdyiw += PActive[k]*dPAk_dy;
                    //}
                } // End if PeVSource

                // Now get ingredients for the calculation of dU_IR/dx,dy 
                // ------------------------------------------------------
                if (eta_IR!=0. && !usetrueE) {
                    // Compute variation of integrated resolution with respect to dx, dy
                    // We define U_IR = sum_k {PA[k]*Wk} / sum_k {PA[k]*Wk * sigmaE/Et}
                    // with      Wk = 1. + Wslope*log(Et/Emin)
                    // and       deltaEk = sigmaE/Et
                    // where Et is the true shower energy (constant).
                    // The derivative wrt xi is then found as
                    //           dU/dxi = {[num'*den - num*den']/[den^2]}
                    // with
                    //           num  = sum_k {PA[k]*Wk}
                    //           den  = sum_k {PA[k]*Wk*deltaEk}
                    //           den' = sum_k {dPA[k]/dxi*Wk/deltaEk + (NB)
                    //                  PA[k]*Wk*d(1/sigmaE)/dR * dRik/dxi /Et } 
                    //           num' = sum_k {dPA[k]/dxi Wk}
                    //
                    // Note: in principle when you sum on Nbatch, and there is a term Pactive[k], you should
                    // consider _all_ showers - also those that are not active. But we do not have a reconstruction
                    // for those that are inactive. Here it is ok, as we are considering variations of the
                    // resolution of those showers we did reconstruct. But above, where we need to assess how
                    // showers contribute to the U_GF piece (there is a density factor 1/sqrt(Nactive) there),
                    // we need to compute dpactive/dxi and there we want to sum on ALL batch showers, as that
                    // number is accessible also for showers we did not reconstruct.
                    // ---------------------------------------------------------------------------------------------
                    
                    double E_dE  = TrueE[k]*InvRmsE[k];
                    //cout << " e_de is " << E_dE << " " << TrueE[k] << " " << InvRmsE[k] << endl;
                    if (fabs(SumProbGe1[k]-Ntrigger)<SumProbRange) { 
                        sumdPAkdxidE2 += dPAk_dx * Wk / E_dE; 
                        sumdPAkdyidE2 += dPAk_dy * Wk / E_dE; 
                        if (sumdPAkdxidE2!=sumdPAkdxidE2 || sumdPAkdyidE2!=sumdPAkdyidE2) {
                            cout    << "sumdPAkdide2 nan: " << dPAk_dx << " " << dPAk_dy << " " << Wk << " " << E_dE << endl;
                            outfile << "sumdPAkdide2 nan: " << dPAk_dx << " " << dPAk_dy << " " << Wk << " " << E_dE << endl;
#if defined(STANDALONE) || defined(UBUNTU)
                            datamutex.lock();
#endif
                            warnings3++; 
#if defined(STANDALONE) || defined(UBUNTU)
                            datamutex.unlock();
#endif
                            TerminateAbnormally();
                        }
                    }
                    // Note the sign of the term 
                    // -------------------------
                    // The denominator is Sum[P*W/(invs_i*E_i)] so when we derive by dinvs/dr we get
                    // d(den)/dinvs = -PW/(invs^2*E)*d(invs)/dR
                    // -----------------------------------------------------------------------------
                    if (InvRmsE[k]==0.) {
                        cout    << "invrms = " << InvRmsE[k] << " and inv_rms_E[k,id,1]=" << inv_rms_E(k,id,1) 
                                << " for k = " << k << endl;
                        outfile << "invrms = " << InvRmsE[k] << " and inv_rms_E[k,id,1]=" << inv_rms_E(k,id,1) 
                                << " for k = " << k << endl;
#if defined(STANDALONE) || defined(UBUNTU)
                        datamutex.lock();
#endif
                        warnings6++; 
#if defined(STANDALONE) || defined(UBUNTU)
                        datamutex.unlock();
#endif
                    }
                    double factor = PActive[k] * Wk * inv_rms_E(k,id,1) / TrueE[k] / (-pow(InvRmsE[k],2.));                    
                    //if (factor>10000000) cout << "factor = " << factor << " " << PActive[k] << " " << Wk << " dinvrmsde=" << inv_rms_E(k,id,1) << " " << TrueE[k] << " " << InvRmsE[k] << endl;
                    sum_dedx += factor*dRdx; 
                    sum_dedy += factor*dRdy;
                    
                    // if (sum_dedx!=sum_dedx) cout << "sumdedx nan " << factor << " " << PActive[k] 
                    //                              << " " << inv_rms_E(k,id,1) << " " << InvRmsE[k] << " " << dRdx << endl;
                    // if (sum_dedy!=sum_dedy) cout << "sumdedy nan " << factor << " " << PActive[k] 
                    //                              << " " << inv_rms_E(k,id,1) << " " << InvRmsE[k] << " " << dRdy << endl;

                    // New attempt, defining denominator as Sum[P*W*de2/et2]
                    // -----------------------------------------------------
                    
                    /*
                    double de2_e2 = pow((TrueE[k]-e_meas[k][0])/TrueE[k],2.);
                    // trim it
                    if (de2_e2>1.) de2_e2 = 1.;
                    if (fabs(SumProbGe1[k]-Ntrigger)<SumProbRange) { 
                        sumdPAkdxidE2 += dPAk_dx * Wk * de2_e2; 
                        sumdPAkdyidE2 += dPAk_dy * Wk * de2_e2; 
                    }
                    double dedr = dEk_dRik(id,k);
                    double factor = PActive[k] * Wk * (-2.*dedr*(TrueE[k]-e_meas[k][0]))/pow(TrueE[k],2.);
                    if (de2_e2>1.) factor = 0.;
                    sum_dedx += factor*dRdx;
                    sum_dedy += factor*dRdy;
                    */

                    //  New definition of utility ir
                    //  ----------------------------
                    // double de_e = fabs(TrueE[k]-e_meas[k][0])/TrueE[k];
                    //  Winsorize it to avoid too much weight from outliers
                    //  ---------------------------------------------------
                    // if (de_e>0.5) de_e = 0.5;
                    // if (fabs(SumProbGe1[k]-Ntrigger)<SumProbRange) { 
                        // sumdPAkdxidE2 += dPAk_dx * Wk * de_e; 
                        // sumdPAkdyidE2 += dPAk_dy * Wk * de_e; 
                    // }
                    // try to see what happens if we assume no direct dependence, by commenting next few lines
                    // double dedr = dEk_dRik(id,k);
                    // double factor = PActive[k] * Wk * dedr/TrueE[k];
                    //if (TrueE[k]>e_meas[k][0]) factor = -factor;
                    //if (de_e>1.) factor = 0.;
                    //sum_dedx += factor*dRdx;
                    //sum_dedy += factor*dRdy;
                }

                // Also get variations for dUPR_dx,dy calculation
                // ----------------------------------------------
                if ((eta_PR!=0. && !usetrueAngs) || PeVSource) {

                    // To get dU_PR/dxi, we need to compute dtheta_meas/dxi and dphi_meas/dxi.
                    // They are obtained by setting to zero dlnL/dtheta (dphi), which is satisfied for the measured solutions,
                    // and deriving with respect to x_i.
                    // Below is a discussion of the derivation, but we are now using Mathematica for the laborious calculations.
                    // ------------------------------------------------------------------------------------------------------------------------------
                    // Previous calculation (with no background), until v98:
                    //
                    //        [*] dlnL/dth = sum(i) [(N/lambda0-costh)dlambda0/dth + sinth(lambda0-N/costh)+ d(log t-term)/dth]
                    // where above we sum on all detectors i, and there is an implicit sum over muons and electrons.
                    // and where the t-term is
                    //        [**] (t-T)/sig_t^2 costh/c(cosph dx + sinph dy)
                    // with
                    //        dx = x_i*-X0, dy = y_i*-Y0
                    //
                    // If we derive wrt i*, all sum terms with i!=i* disappear and we arrive at this expression:
                    //        dtheta/dxi* [sinth dl0/dth + costh l0 - N(1+tgth^2) - tgth (t-term)] = 
                    //        = -{N/l0^2 dl0/dx dl0/dth - costh d^2l0/(dth dx) + sinth dl0/dx + (t-term)[1/(t-T)(-dT/dx)+(cosph+sinph)/(cosph dx + sinph dy)]}
                    // ---------------------------------------------------------------------------------------------------------------------------------------
                    // [Updated solution with background in model, v99 onwards]
                    // We start with 
                    //        L = prod(i) [exp(-lm)*lm^N/N!]_i*[t-term]_i and lm = lm0*ct + TA*Bgrmu, (+ e terms)
                    // so we get 
                    //        log L = sum(i) {-lm + N log lm -log N! +log(t-term)}_i
                    //              = sum(i) {-lm0*ct -TA*Bgrmu + N log (lm0*ct + TA*Bgrmu) -logN! + log(t-term)}_i
                    //        dlnL/dth = dlnL/dth + dlnL/dlm0*dlm0/dth =
                    //                 = sum(i) {st*lm0 -N/(lm0*ct+Ta*Bgrmu)*lm0*st -ct*dlm0/dth +N/(lm0*ct+Ta*Bgrmu)*ct*dlm0/dth +d(log t-term)/dth}_i
                    //                 = sum(i) {[N/(lm0*ct+Ta*Bgrmu)-1]*ct*dlm0/dth + st*lm0*[1-N/(lm0+ct+Ta*Bgrmu)] + dlog(t-term)/dth}_i
                    //                 = sum(i) {[N/(lm0*ct+Ta*Bgrmu)-1]*(ct*dlm0/dth -st*lm0) + dlog(t-term)/dth}_i
                    // As a cross-check, if Bgrmu=0 we would simplify to    
                    //                 = sum(i) {[N/(lm0*ct)-1]*(ct*dlm0/dth-st*lm0) + dlog(t-term)/dth}_i
                    //                 = sum(i) {[N/lm0-ct]*dlm0/dth -st*lm0*[N/(lm0*ct)-1] + dlog(t-term)/dth}_i
                    // which is identical to the earlier expression found earlier (marked [*]).
                    // Here the t-term is more complex because there is a probability f_s that the particle is from signal, with a Gaussian
                    // distribution around true time, and a probability (1-f_s) that it has a flat time distribution, with
                    //        f_s = flux0/(flux0+fluxB) 
                    // The t-term then is now, working back from the likelihood expression: 
                    //        L = prod(i) .... * {f_s*G(t-T,sig_t)+(1-f_s)/[Tmax-Tmin]}_i
                    //        log L = sum(i) {.... + log[f_s*G+(1-f_s)/dT]}_i
                    //        dlnL/dth += sum(i) {[(G-1/dT)*df_s/dth + f_s*dG/dth]/[f_s*G+(1-f_s)/dT]}_i
                    // In the above expression the ingredients are:
                    //        G = 1/sqrt(2*pi*sig_t^2)*exp(-0.5*pow((t-T)/sig_t,2))
                    // For the derivative of f_s, and for any variable x, we have:
                    //        f_s = flux0/(flux0+fluxB) = 1 - fluxB/(flux0+fluxB)
                    //        df_s/dx = fluxB/(flux0+fluxB)^2 * dflux0/dx 
                    // which we can extract from the original derivatives of the flux, computed for mode!=0.  
                    // The above has an impact in the timing distribution and the derivatives, used e.g. in shower reconstruction likelihood:
                    //        P(tmu) = G(tmu,sigma)*f_s + (1-f_s)/T_range
                    // where T_range is the time range we want to give to background muon counts. Taking the log and deriving wrt theta we get: 
                    //        logP(tmu) = log [G(tmu,sigma)*f_s +(1-f_s)/T_range]
                    //        dlogP(tmu)/dth = {dG(tmu,sigma)/dth * f_s + [G(tmu,sigma)-1/T_range] * df_s/dth} / [G(tmu,sigma)*f_s + (1-f_s)/T_range]
                    //                       = {[**] + [G(tmu,sigma)-1/T_range] * fluxB/(flux0+fluxB)^2 dflux0/dth} / [G*f_s + (1-f_s)/T_range]
                    //
                    // ---------------------------------------------------------------------------------------------------------------------------------------
                    // The above can be solved for dtheta/dx if we compute:
                    //        dl0/dx, d/dx(dl0/dth), dT/dx.
                    // For the first we use the already available calculations in the routines;
                    // for the second term, we compute it as
                    //        d/dx(dl0/dth) = d/dx[flux*(1/p0 dp0/dth -p2 R^(p2-1)dR/dth -R^p2 logR dp2/dth)]       
                    // and we have respectively:
                    //        d/dx(dp0/dth) = 0 (I neglect the dependence of this through E and theta)
                    //        d/dx(-flux p2 R^(p2-1)dR/dth) =-flux p2 R^(p2-1) [ logR dR/dx dR/dth - d/dx(dR/dth)] 
                    // Now we get
                    //        d/dx(dR/dth) = d/dx[(-sinthcosph dx - sinth sinph dy)*costh/R(dx cosph+dy sinph)]
                    //        = 0
                    //        d/dx(-flux R^p2 logR dp2/dth) = -flux dp2/dth R^(p2-1)*[(logR)^2+R]dR/dx
                    //
                    // Concerning dphi/dx, we have the following:
                    //        dlnL/dph = sum(i) [dR/dphi*dlogL/dR + (tm-T)/sigma_t^2 sinth/c (-sinph dx + cosp dy)]
                    // where 
                    //        dr/dphi =  -t sinth (-dx sinph + dycosph)/R
                    //        dlogL/dR = (Nm/lambdamu - costh) dlambdamu/dR + e-term
                    //
                    // We now may derive dlnL/dph with respect to xi to harvest the dphi/dxi value. We need these terms:
                    //        d/dxi(dR/dphi) = -sinth^2/R {dphi/dx[(dx^2-dy^2)(-2cosph^2+1)-4dxdy sinph cosph] + (2cosph^2-1)dy -2dx sinph cosph}.
                    //        d/dxi(dlogL/dR) = (-Nmu/lambdamu^2 dlambdamu/dxi dlambdamu/dR + (Nmu/lambdamu -costh) d^2 lambdamu/dxdR)) + e-terms
                    //        d/dxi [(tmu-T)/sigma_t^2 sinth/c (-sinph dx + cosph dy)] + e-terms = 
                    //                = -sinth/(c sigma_t^2) [ dT/dxi (-dx sinph + dy cosph) + (tmu-T)(sinph + dphi/dx (cosph dx + sinph dy))] + e-terms
                    // We thus arrive at the following expression for dphi/dx:
                    //        0 = dphi/dx [-sinth^2/R {(dx^2-dy^2)(-2cosph^2+1)-4dxdy sinph cosph}*dlogL/dR -sinth/(c sigma_t^2)(tmu-T)(cosph dx+sinph dy)] +
                    //            {sinth^2/R [(2cosph^2-1)dy-2dx sinph cosph]*dlogL/dR + dR/dphi d/dx(dlogL/dR) 
                    //             -sinth/(c sigma_t^2) [dT/dx (-dx sinph + dy cosph) + (tmu-T) sinph]}
                    // From this we derive dphi/dx. 
                    // ---------------------------------------------------------------------------------------------------------------------------------------            
                    int nm           = Nm[id][k];
                    int ne           = Ne[id][k]; 
                    float tm         = Tm[id][k];
                    float te         = Te[id][k];
                    double dx        = xi - x0m;
                    double dy        = yi - y0m;
                    double ct        = cos(thm);
                    double st        = sin(thm);
                    double tt        = st/ct;
                    double cp        = cos(phm);
                    double sp        = sin(phm);
                    double thisR     = EffectiveDistance (xi, yi, x0m, y0m, thm, phm, 0);
                    double tmpt      = EffectiveTime (xi, yi, x0m, y0m, thm, phm, 0);
                    double lM0       = MFromG (Ek, thm, thisR, 0);
                    double lE0       = EFromG (Ek, thm, thisR, 0);
                    double fbgm      = TankArea * Bgr_mu_per_m2/(lM0*ct+fluxB_mu);
                    double fbge      = TankArea * Bgr_e_per_m2/(lE0*ct+fluxB_e);
                    if (lM0*lE0==0.) {
                        cout    << "Warning, zero fluxes? Skipping this photon. lm,le = " << lM0 << " " << lE0 << " Emeas = " << e_meas[k][0] << " R = " << thisR << endl;
                        outfile << "Warning, zero fluxes? Skipping this photon. lm,le = " << lM0 << " " << lE0 << " Emeas = " << e_meas[k][0] << " R = " << thisR << endl;
#if defined(STANDALONE) || defined(UBUNTU)
                        datamutex.lock();
#endif
                        warnings7++; 
#if defined(STANDALONE) || defined(UBUNTU)
                        datamutex.unlock();
#endif
                    } else {
                        double dRdTh       = EffectiveDistance (xi, yi, x0m, y0m, thm, phm, 3);
                        double dR2dTh2     = EffectiveDistance (xi, yi, x0m, y0m, thm, phm, 32);
                        double dlM0dth     = MFromG (Ek, thm, thisR, 3, dRdTh);
                        double dlE0dth     = EFromG (Ek, thm, thisR, 3, dRdTh);
                        double d2lM0dth2   = MFromG (Ek, thm, thisR, 3, dRdTh,dR2dTh2);
                        double d2lE0dth2   = EFromG (Ek, thm, thisR, 3, dRdTh,dR2dTh2);
                        // Below, dlM0dx = dlM0dR * dRdx (dRdx = dx/R if we derive by xi, -dx/R if we derive by x0)
                        double dlM0dR      = MFromG (Ek, thm, thisR, 1);
                        double d2lM0_dthdR = MFromG (Ek, thm, thisR, 31);
                        double dlM0dx      = dlM0dR * dRdx;  
                        double dlM0dy      = dlM0dR * dRdy;  
                        double d2lM0_dthdx = d2lM0_dthdR * dRdx;
                        double d2lM0_dthdy = d2lM0_dthdR * dRdy;
                        double dlE0dR      = EFromG (Ek, thm, thisR, 1);
                        double d2lE0_dthdR = EFromG (Ek, thm, thisR, 31);
                        double dlE0dx      = dlE0dR * dRdx; 
                        double dlE0dy      = dlE0dR * dRdy;
                        double d2lE0_dthdx = d2lE0_dthdR * dRdx;
                        double d2lE0_dthdy = d2lE0_dthdR * dRdy;

                        //double s      = sigma_time;
                        double s3     = pow(sigma_time,3.);
                        double s5     = pow(sigma_time,5.);
                        double a      = TankArea;
                        double bm     = fluxB_mu;
                        double be     = fluxB_e;
                        double r      = IntegrationWindow;
                        double denm   = bm+ct*lM0;
                        double denm2  = denm*denm; 
                        double denm3  = denm2*denm;
                        double dene   = be+ct*lE0;
                        double dene2  = dene*dene; 
                        double dene3  = dene2*dene;
                        double fm     = bm/denm;
                        double fe     = be/dene;
                        double t      = (cp*dx+sp*dy)*st/c0; // Expected time given the shower front and measured coordinates
                        double Gm     = exp(-0.5*pow((tm-t)/sigma_time,2.));
                        double Ge     = exp(-0.5*pow((te-t)/sigma_time,2.));
                        double dth_dx = 0.;
                        double dth_dy = 0.;
                        double dph_dx = 0.;
                        double dph_dy = 0.;
                        // The condition below is imposed to avoid irrelevant calculations. 
                        // NB for det id not seeing e or m, te and tm were set to zero, which meant most Ge,Gm were zero (but not all). Do we want this?
                        // -----------------------------------------------------------------------------------------------------------------------------
                        if (Gm>epsilon2 && Ge>epsilon2) { 
                            double s2p   = sqrt2pi;
                            double mdm   = -lM0*st+dlM0dth*ct;
                            double ede   = -lE0*st+dlE0dth*ct;
                            double css3  = c0*s2p*s3;
                            double css5  = c0*s2p*s5;
                            double gm    = fm/r + (1-fm)*Gm/(s2p*sigma_time);
                            double gm2   = gm*gm;
                            double ge    = fe/r + (1-fe)*Ge/(s2p*sigma_time);
                            double ge2   = ge*ge;
                            double tmpdf = fm/IntegrationWindow + (1-fm)*Gm/(sigma_time*sqrt2pi);
                            double tmpdf2= tmpdf*tmpdf;
                            double tepdf = fe/IntegrationWindow + (1-fe)*Ge/(sigma_time*sqrt2pi);
                            double tepdf2= tepdf*tepdf;
                            double tmt   = tm-t;
                            double tmt2  = pow(tmt,2.);
                            double tet   = te-t;
                            double tet2  = pow(tet,2.);
                            double k3e   = (1-fe)*Ge*t*tet/(css3) - be*ede/(r*dene2) + be*Ge*ede/(s2p*sigma_time*dene2);                                       
                            double k3m   = (1-fm)*Gm*t*tmt/(css3) - bm*mdm/(r*denm2) + bm*Gm*mdm/(s2p*sigma_time*denm2);

                            double denom_css3_ge = css3 * ge;
                            double denom_css3_gm = css3 * gm;
                            double denom_css5_ge = css5 * ge;
                            double denom_css5_gm = css5 * gm;
                            double denom_css3_ge2 = denom_css3_ge * ge;
                            double denom_css3_gm2 = denom_css3_gm * gm;
                            double denom_css3_dene2_ge = css3 * dene2 * ge;
                            double denom_css3_denm2_gm = css3 * denm2 * gm;                            
                            double denom_s2p_s_dene2_ge = s2p * sigma_time * dene2 * ge;
                            double denom_s2p_s_denm2_gm = s2p * sigma_time * denm2 * gm;
                            double denom_s2p_s_dene2_ge2 = denom_s2p_s_dene2_ge * ge;
                            double denom_s2p_s_denm2_gm2 = denom_s2p_s_denm2_gm * gm;
                            double denom_r_dene2_ge = r * dene2 * ge;
                            double denom_r_denm2_gm = r * denm2 * gm;
                            double denom_r_dene2_ge2 = denom_r_dene2_ge * ge;
                            double denom_r_denm2_gm2 = denom_r_denm2_gm * gm;
                            double denom_r_dene3_ge = denom_r_dene2_ge * dene;
                            double denom_r_denm3_gm = denom_r_denm2_gm * denm;
                            double denom_s2p_s3_dene2_ge = s2p * s3 * dene2 * ge;
                            double denom_s2p_s3_denm2_gm = s2p * s3 * denm2 * gm;
                            double denom_s2p_s_dene3_ge = denom_s2p_s_dene2_ge * dene;
                            double denom_s2p_s_denm3_gm = denom_s2p_s_denm2_gm * denm; 

                            double bignumx, bigdenx, bignumy, bigdeny;
                            if (ne+nm>0) { 
                                bignumx =       + t * cp * (1. - fe) * Ge * st / denom_css3_ge
                                                + t * cp * (1. - fm) * Gm * st / denom_css3_gm
                                                - ct * cp * (1. - fe) * Ge * tet / denom_css3_ge
                                                - be * cp * ede * Ge * st * tet / denom_css3_dene2_ge
                                                + cp * (1. - fe) * Ge * k3e * st * tet / denom_css3_ge2
                                                - t * cp * (1. - fe) * Ge * st * tet2 / denom_css5_ge
                                                - cp * ct * (1. - fm) * Gm * tmt / denom_css3_gm
                                                - bm * cp * Gm * st * tmt * mdm / denom_css3_denm2_gm
                                                + cp * (1. - fm) * Gm * k3m * st * tmt / denom_css3_gm2
                                                - t * cp * (1. - fm) * Gm * st * tmt2 / denom_css5_gm
                                                - 2. * be * ct * ede * dlE0dx / denom_r_dene3_ge
                                                + 2. * be * ct * ede * Ge * dlE0dx / denom_s2p_s_dene3_ge
                                                - 2. * bm * ct * mdm * dlM0dx / denom_r_denm3_gm
                                                + 2. * bm * ct * Gm * mdm * dlM0dx / denom_s2p_s_denm3_gm
                                                - be * ct * k3e * dlE0dx / denom_r_dene2_ge2
                                                + be * ct * Ge * k3e * dlE0dx / denom_s2p_s_dene2_ge2
                                                - bm * ct * k3m * dlM0dx / denom_r_denm2_gm2
                                                + bm * ct * Gm * k3m * dlM0dx / denom_s2p_s_denm2_gm2
                                                - be * st * dlE0dx / denom_r_dene2_ge
                                                + be * Ge * st * dlE0dx / denom_s2p_s_dene2_ge
                                                - bm * st * dlM0dx / denom_r_denm2_gm
                                                + bm * Gm * st * dlM0dx / denom_s2p_s_denm2_gm
                                                + be * ct * d2lE0_dthdx / denom_r_dene2_ge
                                                - be * ct * Ge * d2lE0_dthdx / denom_s2p_s_dene2_ge
                                                + bm * ct * d2lM0_dthdx / denom_r_denm2_gm
                                                - bm * ct * Gm * d2lM0_dthdx / denom_s2p_s_denm2_gm
                                                + ct * ede * ne * dlE0dx / dene2
                                                + ct * nm * mdm * dlM0dx / denm2
                                                - a * st * dlE0dx
                                                - a * st * dlM0dx
                                                + ne * st * dlE0dx / dene
                                                - be * t * ct * Ge * tet * dlE0dx / denom_s2p_s3_dene2_ge
                                                + nm * st * dlM0dx / denm
                                                - bm * t * ct * Gm * tmt * dlM0dx / denom_s2p_s3_denm2_gm
                                                + a * ct * d2lE0_dthdx
                                                - ct * ne * d2lE0_dthdx / dene
                                                + a * ct * d2lM0_dthdx
                                                - ct * nm * d2lM0_dthdx / denm;

                                bigdenx =       - dx * t * ct * cp * (1. - fe) * Ge / denom_css3_ge
                                                - dx * t * ct * cp * (1. - fm) * Gm / denom_css3_gm
                                                - dy * t * ct * (1. - fe) * Ge * sp / denom_css3_ge
                                                - dy * t * ct * (1. - fm) * Gm * sp / denom_css3_gm
                                                + dx * be * ct * cp * ede * Ge * tet / denom_css3_dene2_ge
                                                + dy * be * ct * sp * ede * Ge * tet / denom_css3_dene2_ge
                                                + dx * bm * ct * cp * Gm * tmt * mdm / denom_css3_denm2_gm
                                                + dy * bm * ct * sp * Gm * tmt * mdm / denom_css3_denm2_gm
                                                - dx * ct * cp * (1. - fe) * Ge * k3e * tet / denom_css3_ge2
                                                - dy * ct * sp * (1. - fe) * Ge * k3e * tet / denom_css3_ge2
                                                - dx * ct * cp * (1. - fm) * Gm * k3m * tmt / denom_css3_gm2
                                                - dy * ct * sp * (1. - fm) * Gm * k3m * tmt / denom_css3_gm2
                                                - dx * cp * (1. - fe) * Ge * st * tet / denom_css3_ge
                                                - dy * sp * (1. - fe) * Ge * st * tet / denom_css3_ge
                                                - dx * cp * (1. - fm) * Gm * st * tmt / denom_css3_gm
                                                - dy * sp * (1. - fm) * Gm * st * tmt / denom_css3_gm
                                                + dx * t * ct * cp * (1. - fe) * Ge * tet2 / denom_css5_ge
                                                + dy * t * ct * sp * (1. - fe) * Ge * tet2 / denom_css5_ge
                                                + dx * t * ct * cp * (1. - fm) * Gm * tmt2 / denom_css5_gm
                                                + dy * t * ct * sp * (1. - fm) * Gm * tmt2 / denom_css5_gm
                                                + a * ct * lE0 
                                                + be * ct * lE0 / denom_r_dene2_ge
                                                - be * ct * Ge * lE0 / denom_s2p_s_dene2_ge
                                                - ct * ne * lE0 / dene
                                                + a * ct * lM0 
                                                + bm * ct * lM0 / denom_r_denm2_gm
                                                - bm * ct * Gm * lM0 / denom_s2p_s_denm2_gm
                                                - ct * nm * lM0 / denm
                                                - 2. * be * ede * st * lE0 / denom_r_dene3_ge
                                                + 2. * be * ede * Ge * st * lE0 / denom_s2p_s_dene3_ge
                                                - 2. * bm * st * lM0 * mdm / denom_r_denm3_gm
                                                + 2. * bm * Gm * st * lM0 * mdm / denom_s2p_s_denm3_gm
                                                - be * k3e * st * lE0 / denom_r_dene2_ge2
                                                + be * Ge * k3e * st * lE0 / denom_s2p_s_dene2_ge2
                                                - bm * k3m * st * lM0 / denom_r_denm2_gm2
                                                + bm * Gm * k3m * st * lM0 / denom_s2p_s_denm2_gm2
                                                + ede * ne * st * lE0 / dene2
                                                + nm * st * lM0 * mdm / denm2
                                                - be * t * Ge * st * tet * lE0 / denom_s2p_s3_dene2_ge
                                                - bm * t * Gm * st * tmt * lM0 / denom_s2p_s3_denm2_gm
                                                + 2. * be * ct * ede * dlE0dth / denom_r_dene3_ge
                                                - 2. * be * ct * ede * Ge * dlE0dth / denom_s2p_s_dene3_ge
                                                + 2. * bm * ct * mdm * dlM0dth / denom_r_denm3_gm
                                                - 2. * bm * ct * Gm * mdm * dlM0dth / denom_s2p_s_denm3_gm
                                                + be * ct * k3e * dlE0dth / denom_r_dene2_ge2
                                                - be * ct * Ge * k3e * dlE0dth / denom_s2p_s_dene2_ge2
                                                + bm * ct * k3m * dlM0dth / denom_r_denm2_gm2
                                                - bm * ct * Gm * k3m * dlM0dth / denom_s2p_s_denm2_gm2
                                                - ct * ede * ne * dlE0dth / dene2
                                                - ct * nm * mdm * dlM0dth / denm2
                                                + 2. * a * st * dlE0dth
                                                + 2. * a * st * dlM0dth
                                                + 2. * be * st * dlE0dth / denom_r_dene2_ge
                                                + 2. * bm * st * dlM0dth / denom_r_denm2_gm
                                                - 2. * be * Ge * st * dlE0dth / denom_s2p_s_dene2_ge
                                                - 2. * bm * Gm * st * dlM0dth / denom_s2p_s_denm2_gm
                                                - 2. * ne * st * dlE0dth / dene
                                                - 2. * nm * st * dlM0dth / denm
                                                + be * t * ct * Ge * tet * dlE0dth / denom_s2p_s3_dene2_ge
                                                + bm * t * ct * Gm * tmt * dlM0dth / denom_s2p_s3_denm2_gm
                                                - a * ct * d2lE0dth2
                                                - be * ct * d2lE0dth2 / denom_r_dene2_ge
                                                + be * ct * Ge * d2lE0dth2 / denom_s2p_s_dene2_ge
                                                + ct * ne * d2lE0dth2 / dene
                                                - a * ct * d2lM0dth2
                                                - bm * ct * d2lM0dth2 / denom_r_denm2_gm
                                                + bm * ct * Gm * d2lM0dth2 / denom_s2p_s_denm2_gm
                                                + ct * nm * d2lM0dth2 / denm;

                                bignumy          =  + t * sp * (1. - fe) * Ge * st / denom_css3_ge
                                                    + t * sp * (1. - fm) * Gm * st / denom_css3_gm
                                                    - ct * sp * (1. - fe) * Ge * tet / denom_css3_ge
                                                    - be * sp * ede * Ge * st * tet / denom_css3_dene2_ge
                                                    + sp * (1. - fe) * Ge * k3e * st * tet / denom_css3_ge2
                                                    - t * sp * (1. - fe) * Ge * st * tet2 / denom_css5_ge
                                                    - sp * ct * (1. - fm) * Gm * tmt / denom_css3_gm
                                                    - bm * sp * Gm * st * tmt * mdm / denom_css3_denm2_gm
                                                    + sp * (1. - fm) * Gm * k3m * st * tmt / denom_css3_gm2
                                                    - t * sp * (1. - fm) * Gm * st * tmt2 / denom_css5_gm
                                                    - 2. * be * ct * ede * dlE0dy / denom_r_dene3_ge
                                                    + 2. * be * ct * ede * Ge * dlE0dy / denom_s2p_s_dene3_ge
                                                    - 2. * bm * ct * mdm * dlM0dy / denom_r_denm3_gm
                                                    + 2. * bm * ct * Gm * mdm * dlM0dy / denom_s2p_s_denm3_gm
                                                    - be * ct * k3e * dlE0dy / denom_r_dene2_ge2
                                                    + be * ct * Ge * k3e * dlE0dy / denom_s2p_s_dene2_ge2
                                                    - bm * ct * k3m * dlM0dy / denom_r_denm2_gm2
                                                    + bm * ct * Gm * k3m * dlM0dy / denom_s2p_s_denm2_gm2
                                                    - be * st * dlE0dy / denom_r_dene2_ge
                                                    + be * Ge * st * dlE0dy / denom_s2p_s_dene2_ge
                                                    - bm * st * dlM0dy / denom_r_denm2_gm
                                                    + bm * Gm * st * dlM0dy / denom_s2p_s_denm2_gm
                                                    + be * ct * d2lE0_dthdy / denom_r_dene2_ge
                                                    - be * ct * Ge * d2lE0_dthdy / denom_s2p_s_dene2_ge
                                                    + bm * ct * d2lM0_dthdy / denom_r_denm2_gm
                                                    - bm * ct * Gm * d2lM0_dthdy / denom_s2p_s_denm2_gm
                                                    + ct * ede * ne * dlE0dy / dene2
                                                    + ct * nm * mdm * dlM0dy / denm2
                                                    - a * st * dlE0dy
                                                    - a * st * dlM0dy
                                                    + ne * st * dlE0dy / dene
                                                    - be * t * ct * Ge * tet * dlE0dy / denom_s2p_s3_dene2_ge
                                                    + nm * st * dlM0dy / denm
                                                    - bm * t * ct * Gm * tmt * dlM0dy / denom_s2p_s3_denm2_gm
                                                    + a * ct * d2lE0_dthdy
                                                    - ct * ne * d2lE0_dthdy / dene
                                                    + a * ct * d2lM0_dthdy
                                                    - ct * nm * d2lM0_dthdy / denm;

                                bigdeny = bigdenx;

                            } else { 
                                // If nm+ne=0 the likelihood for this detector is MUCH simpler. 
                                // See notebook: Solution dtheta_dx with background in flux e and mu for n=0.nb
                                // ----------------------------------------------------------------------------
                                bignumx = -st*(dlE0dx+dlM0dx) +ct*(d2lE0_dthdx+d2lM0_dthdx);
                                bigdenx = ct*(lE0+lM0) + 2*st*(dlE0dth+dlM0dth) -ct*(d2lE0dth2+d2lM0dth2);
                                bignumy = -st*(dlE0dy+dlM0dy) +ct*(d2lE0_dthdy+d2lM0_dthdy);
                                bigdeny = ct*(lE0+lM0) + 2*st*(dlE0dth+dlM0dth) -ct*(d2lE0dth2+d2lM0dth2);
                            }

                            if (bigdenx==0) {
                                cout    << "Warning, denominator null in dthx calculation " << endl;                            
                                outfile << "Warning, denominator null in dthx calculation " << endl;                            
#if defined(STANDALONE) || defined(UBUNTU)
                                datamutex.lock();
#endif
                                warnings5++; 
#if defined(STANDALONE) || defined(UBUNTU)
                                datamutex.unlock();
#endif
                            } else {
                                dth_dx = bignumx/bigdenx;
                            }
                            if (bigdeny==0) {
                                cout    << "Warning, denominator null in dthy calculation " << endl;
                                outfile << "Warning, denominator null in dthy calculation " << endl;
#if defined(STANDALONE) || defined(UBUNTU)
                                datamutex.lock();
#endif
                                warnings5++; 
#if defined(STANDALONE) || defined(UBUNTU)
                                datamutex.unlock();
#endif
                            } else {
                                dth_dy = bignumy/bigdeny;
                            }
                            if (dth_dx!=dth_dx) {
                                cout    << "Warniing, trouble with dth/dx:" << endl; 
                                cout    << ge << " " << gm << " " << denm << " " << dene << " " << Ge << " " << Gm << endl; 
                                outfile << "Warning, trouble with dth/dx:" << endl; 
                                outfile << ge << " " << gm << " " << denm << " " << dene << " " << Ge << " " << Gm << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
                                datamutex.lock();
#endif
                                warnings5++; 
#if defined(STANDALONE) || defined(UBUNTU)
                                datamutex.unlock();
#endif
                                dth_dx = 0;
                            }
                            if (dth_dy!=dth_dy) {
                                cout    << "Warning, trouble with dth/dy:" << endl; 
                                cout    << ge << " " << gm << " " << denm << " " << dene << " " << Ge << " " << Gm << endl; 
                                outfile << "Warning, trouble with dth/dy:" << endl; 
                                outfile << ge << " " << gm << " " << denm << " " << dene << " " << Ge << " " << Gm << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
                                datamutex.lock();
#endif
                                warnings5++; 
#if defined(STANDALONE) || defined(UBUNTU)
                                datamutex.unlock();
#endif
                                dth_dy = 0;
                            }
                            //numx->Fill(log(fabs(bignumx)));
                            //numy->Fill(log(fabs(bignumy)));
                            //denx->Fill(log(fabs(bigdenx)));
                            //deny->Fill(log(fabs(bigdeny)));
                            // if (dth_dy!=0) cout << " dth_dx,dy " << dth_dx/dth_dy;
                            // if (dph_dy!=0) cout << " dph_dx,dy " << dph_dx/dph_dy << endl;

                            if (nm+ne>0) { // otherwise there is no phi dependence at all, as there is no time factor in the likelihood
                                // Now compute the phi dependence 
                                // ------------------------------
                                double s6 = s3*s3;
                                double s4 = s3*sigma_time;
                                double fbe = 1.-fe;
                                double fbm = 1.-fm;
                                double fbe2 = fbe*fbe;
                                double fbm2 = fbm*fbm;
                                double ke = fe/r+fbe*Ge/(s2p*sigma_time);
                                double ke2 = ke*ke;
                                double km = fm/r+fbm*Gm/(s2p*sigma_time);
                                double km2 = km*km;
                                double Gm2 = Gm*Gm;
                                double Ge2 = Ge*Ge;
                                double cts6= c0*twopi*s6;
                                double Gest_c3ke = fbe*Ge*st/(css3*ke);
                                double Gmst_c3km = fbm*Gm*st/(css3*km);
                                double bignumpx = t*cp*Gest_c3ke + t*cp*Gmst_c3km + Gest_c3ke*sp*tet -
                                                t*cp*fbe2*Ge2*st*tet2/(cts6*ke2) + t*cp*fbe*Ge*st*tet2/(css5*ke) + Gmst_c3km*sp*tmt +
                                                t*cp*fbm2*Gm2*st*tmt2/(cts6*km2) - t*cp*fbm*Gm*st*tmt2/(css5*km) - be*t*ct*fbe*Ge*tet*dlE0dx/(r*s2p*s3*dene2*ke2) +
                                                be*t*ct*fbe*Ge2*tet*dlE0dx/(twopi*s4*dene2*ke2) - be*t*ct*Ge*tet*dlE0dx/(s2p*s3*dene2*ke) -
                                                bm*t*ct*fbm*Gm*tmt*dlM0dx/(r*s2p*s3*km2*denm2) + bm*t*ct*fbm*Gm2*tmt*dlM0dx/(twopi*s4*km2*denm2) -
                                                bm*t*ct*Gm*tmt*dlM0dx/(s2p*s3*km*denm2);
                                double bigdenpx = -dy*t*cp*Gest_c3ke - dy*t*cp*Gmst_c3km + dx*t*Gest_c3ke*sp +
                                                dx*t*Gmst_c3km*sp - dx*cp*Gest_c3ke*tet - dy*Gest_c3ke*sp*tet -
                                                dy*t*cp*fbe2*Ge2*st*tet2/(cts6*ke2) + dy*t*cp*fbe*Ge*st*tet2/(css5*ke) + 
                                                dx*t*fbe2*Ge2*st*sp*tet2/(cts6*ke2) - dx*t*fbe*Ge*st*sp*tet2/(css5*ke) -
                                                dx*cp*Gmst_c3km*tmt - dy*Gmst_c3km*sp*tmt - dy*t*cp*fbm2*Gm2*st*tmt2/(cts6*km2) +
                                                dy*t*cp*fbm*Gm*st*tmt2/(css5*km) + dx*t*fbm2*Gm2*st*sp*tmt2/(cts6*km2) - dx*t*fbm*Gm*st*sp*tmt2/(css5*km);                                          
                                double bignumpy = t*sp*Gest_c3ke + t*sp*Gmst_c3km - Gest_c3ke*cp*tet +
                                                t*sp*fbe2*Ge2*st*tet2/(cts6*ke2) - t*sp*fbe*Ge*st*tet2/(css5*ke) - Gmst_c3km*cp*tmt +
                                                t*sp*fbm2*Gm2*st*tmt2/(cts6*km2) - t*sp*fbm*Gm*st*tmt2/(css5*km) - be*t*ct*fbe*Ge*tet*dlE0dy/(r*s2p*s3*dene2*ke2) +
                                                be*t*ct*fbe*Ge2*tet*dlE0dy/(twopi*s4*dene2*ke2) - be*t*ct*Ge*tet*dlE0dy/(s2p*s3*dene2*ke) -
                                                bm*t*ct*fbm*Gm*tmt*dlM0dy/(r*s2p*s3*km2*denm2) + bm*t*ct*fbm*Gm2*tmt*dlM0dy/(twopi*s4*km2*denm2) -
                                                bm*t*ct*Gm*tmt*dlM0dy/(s2p*s3*km*denm2);
                                double bigdenpy = bigdenpx; // See mathematica files Solution_dphi_dx... and Solution_dphi_dy
                                if (bigdenpx==0.) {
                                    cout    << "Warning, denominator null in dphx calculation " << endl;                             
                                    outfile << "Warning, denominator null in dphx calculation " << endl;                             
#if defined(STANDALONE) || defined(UBUNTU)
                                    datamutex.lock();
#endif
                                    warnings5++; 
#if defined(STANDALONE) || defined(UBUNTU)
                                    datamutex.unlock();
#endif
                                } else {
                                    dph_dx = bignumpx/bigdenpx;
                                }
                                if (bigdenpy==0.) {
                                    cout    << "Warning, denominator null in dphy calculation " << endl;
                                    outfile << "Warning, denominator null in dphy calculation " << endl;
#if defined(STANDALONE) || defined(UBUNTU)
                                    datamutex.lock();
#endif
                                    warnings5++; 
#if defined(STANDALONE) || defined(UBUNTU)
                                    datamutex.unlock();
#endif
                                } else {
                                    dph_dy = bignumpy/bigdenpy;
                                }
                                if (dph_dx!=dph_dx) {
                                    cout    << "Trouble with dph/dx:" << endl; 
                                    cout    << ke << " " << km << " " << denm << " " << dene << " " << Ge << " " << Gm << endl; 
                                    outfile << "Trouble with dph/dx:" << endl; 
                                    outfile << ke << " " << km << " " << denm << " " << dene << " " << Ge << " " << Gm << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
                                    datamutex.lock();
#endif
                                    warnings5++; 
#if defined(STANDALONE) || defined(UBUNTU)
                                    datamutex.unlock();
#endif
                                    dph_dx = 0.;
                                }
                                if (dph_dy!=dph_dy) {
                                    cout    << "Trouble with dph/dy:" << endl; 
                                    cout    << ke << " " << km << " " << denm << " " << dene << " " << Ge << " " << Gm << endl; 
                                    outfile << "Trouble with dph/dy:" << endl; 
                                    outfile << ke << " " << km << " " << denm << " " << dene << " " << Ge << " " << Gm << endl; 
#if defined(STANDALONE) || defined(UBUNTU)
                                    datamutex.lock();
#endif
                                    warnings5++; 
#if defined(STANDALONE) || defined(UBUNTU)
                                    datamutex.unlock();
#endif
                                    dph_dy = 0.;
                                }
                            } else { // If nm+ne==0 there is no time dependence in phi
                                dph_dx = 0.;
                                dph_dy = 0.;
                            }
                        } // End if Ge*Gm!=0

                        // Finally, collect information in numerator and denominator derivatives
                        // ---------------------------------------------------------------------
                        double weight   = PActive[k] * Wk; 
                        double dtheta   = tht-thm;
                        double dphi     = pi-fabs(fabs(pht-phm)-pi);
                        float ddphi_dpm = -1.;
                        double dphipr   = pi-fabs(fabs(pht-phm-deltapr2)-pi); // we increment phm to see if dphi increases or decreases
                        if (dphipr>dphi) ddphi_dpm = 1.; // Inverts the default sign of the derivative of (pht-phm)
                        double dr       = sqrt(pow(dtheta,2.) + pow(sin(tht)*dphi,2.)+deltapr2);
                        // Note: if we define dtheta = theta_t - theta_m, then when we derive
                        // 1/(dtheta^2+..) over dx we have to do d(1/(dtheta^2+..))/d(dtheta) * d(dtheta)/dx
                        // and the last term brings a negative sign: d(dtheta)/dx = -dtheta_m/dx.
                        // Otherwise if we had defined dtheta = theta_m-theta_t, then we would have 
                        // d(dtheta)/dx = dtheta_m/dx with positive sign.
                        // In the end we get -2*dtheta/den^2 * (-dtheta_m/dx) = +2*dtheta/den^2 dtheta_m/dx
                        // where the numerator has dtheta = theta_t-theta_m: if theta_m<theta_t this is larger
                        // than zero, and the derivative increases the function along the direction of increase
                        // of dtheta_m/dx, as the denominator becomes smaller (theta_m goes closer to theta_t).
                        // ------------------------------------------------------------------------------------  
                        // New def: num = sum_k { PA W delta/R }      with R = sqrt(dth^2+(dph*s(th))^2+delta2)
                        // now dnum/dx  = sum_k { PA W d(delta/R)/dR [dR/dth dth/dx + dR/dph dph/dx] }
                        //              = sum_k { PA W (-delta/R^2) [1/2R(-2*deltath+2*st*ct*deltaph^2)dth/dx +1/2R(-2*deltaph) dph/dx]}
                        //              = sum_k { PA W delta / R^3 [(deltath-st*ct*deltaph^2) dth/dx + deltaph dph/dx]}
                        // but careful about the sign of the last term, as d(deltaph)/dph is not necessarily -1!
                        // -------------------------------------------------------------------------------------
                        // Winsorize at 0.1 the residual of the angle measurement
                        // ------------------------------------------------------
                        //  if (fabs(SumProbGe1[k]-Ntrigger)<0.1*Nunits) {
                        if (fabs(SumProbGe1[k]-Ntrigger)<SumProbRange) { // See elsewhere an explanation of the requirement
                            sum_dnumPRdx += dPAk_dx*Wk*deltapr/dr;
                            // dden_dx      += dPAk_dx*Wk; --> see sumdpakdxi
                            sum_dnumPRdy += dPAk_dy*Wk*deltapr/dr; 
                            // dden_dy      += dPAk_dy*Wk; --> see sumdpakdyi
                            //cout << "sumdnumprdx,dy = " << sum_dnumPRdx << " " << dPAk_dx << " " << sum_dnumPRdy << " " << dPAk_dy << " " << Wk << " " << resden << endl;
                        }
                        double tmp = weight*deltapr/pow(dr,3.);
                        // NB below the factor ddphi_dpm was only introduced for the y component. Bug? Or was it correct?
                        // NNBB ddphi_dph is -1 in default mode, following calculation above, so +sin(th)^2*dphi*dph_dx becames -ddphi_dpm*...
                        sum_dnumPRdx += tmp*((dtheta-dphi*dphi*sin(tht)*cos(tht))*dth_dx-ddphi_dpm*sin(tht)*sin(tht)*dphi*dph_dx);
                        sum_dnumPRdy += tmp*((dtheta-dphi*dphi*sin(tht)*cos(tht))*dth_dy-ddphi_dpm*sin(tht)*sin(tht)*dphi*dph_dy);
                        // cout << " + " << weight << " " << 1./resden2 << " " << (dtheta*dth_dx + dphi * dph_dx) << " " << (dtheta*dth_dy + dphi*dph_dy) << endl;
                        // cout << k << " dth, dph dx " << dth_dx << " " << dph_dx << endl; 

                        // Deal with angular part of Point Source utility
                        // Note: minus signs arise when deriving dtheta = tht-thm over thm, and similarly for dphi,
                        // but for dphi we have to be careful as the formula is dphi = pi-abs(abs(pht-phm)-pi) so we
                        // introduce the ddphi_dpm function.
                        // The definition of sigma_theta^2 is
                        //     sigma_theta^2 = sum_i PA_i*dth^2 / sum_i PA_i - [sum_i PA_i*dth / sum_i PA_i]^2 = num1/den - (num2/den)^2 
                        //  So when we derive we have
                        //     dnum1/dxj = sum_i (dPA_i/dxj*dth^2 -2PA_i*dth*dthm/dxj)
                        //     dnum2/dxj = sum_i (dPA_i/dxj*dth-PA_i*dthm/dxj)
                        // -------------------------------------------------------------------------------------------------------------
                        if (PeVSource) {
                            double dtheta2   = pow(dtheta,2.);
                            double dphi2     = pow(dphi,2.);
                            sum_num1s2t     += weight*dtheta2;
                            sum_num1s2p     += weight*dphi2;
                            sum_num2s2t     += weight*dtheta;
                            sum_num2s2p     += weight*dphi;
                            sum_dens2       += weight;
                            sum_dnum1s2t_dx += dPAk_dx*Wk*dtheta2 - 2.*weight*dtheta*dth_dx;
                            sum_dnum1s2p_dx += dPAk_dx*Wk*dphi2   + 2.*weight*dphi*ddphi_dpm*dph_dx;
                            sum_dnum1s2t_dy += dPAk_dy*Wk*dtheta2 - 2.*weight*dtheta*dth_dy;
                            sum_dnum1s2p_dy += dPAk_dy*Wk*dphi2   + 2.*weight*dphi*ddphi_dpm*dph_dy;
                            sum_dnum2s2t_dx += dPAk_dx*Wk*dtheta - weight*dth_dx;
                            sum_dnum2s2p_dx += dPAk_dx*Wk*dphi   + weight*ddphi_dpm*dph_dx;
                            sum_dnum2s2t_dy += dPAk_dy*Wk*dtheta - weight*dth_dy;
                            sum_dnum2s2p_dy += dPAk_dy*Wk*dphi   + weight*ddphi_dpm*dph_dy;
                            // cout << "for det " << id << " p, tm, te, t = " << PActive[k] << " " << tm << " " << te << " " << t << " dthdx, dy = " << dth_dx << " " << dth_dy << " dphdx,dy = " << dph_dx << " " << dph_dy << endl;
                        }
                    } // If not zero fluxes
                } // If pevsource or ...
            } // End if IsGamma
        } // End k loop on batch events ------------------------------------------------------------------

        d_fs_dx *= sigmafs2;
        d_fs_dy *= sigmafs2;

        // Remember to multiply dinvsfs/dx by sigmafs, as the derivative of f(x)^-0.5 is 1/[2*f(x)]* f'(x) 
        // ... and the sum is f'(x) only. Also, the factor of 2 was taken care of above
        // -----------------------------------------------------------------------------------------------
        d_invsigfs_dx = d_invsigfs_dx * MeasFgErr;
        d_invsigfs_dy = d_invsigfs_dy * MeasFgErr;

        double dIR_dxi = 0.;
        double dIR_dyi = 0.;
        if (eta_IR!=0. && !usetrueE) {
            if (U_IR_Den !=0.) {
                //cout << " sumdpakdxde2 = " << sumdPAkdxidE2 << " sumdedx = " << sum_dedx << " " << U_IR_Num << " " << U_IR_Den << " " << sumdPAkdxi << endl;
                dIR_dxi = (-(sumdPAkdxidE2 + sum_dedx) * U_IR_Num + U_IR_Den * sumdPAkdxi) / pow(U_IR_Den,2.);
                dIR_dyi = (-(sumdPAkdyidE2 + sum_dedy) * U_IR_Num + U_IR_Den * sumdPAkdyi) / pow(U_IR_Den,2.);
            } else {
                cout    << "Warning, U_IR_Den is null " << endl;
                outfile << "Warning, U_IR_Den is null " << endl;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings5++; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
            }
            //if (fabs(dIR_dxi)>10000.) cout << "     dirdx prob " << sumdPAkdxidE2 << " " << sum_dedx << " " << sumdPAkdxi << " " << U_IR_Num << " " << U_IR_Den << endl;
            //if (fabs(dIR_dyi)>10000.) cout << "     dirdy prob " << sumdPAkdyidE2 << " " << sum_dedy << " " << sumdPAkdyi << " " << U_IR_Num << " " << U_IR_Den << endl;
        }
        double dPR_dxi = 0.;
        double dPR_dyi = 0.;
        if (eta_PR!=0. && !usetrueAngs) {
            if (U_PR_Den !=0.) {
                dPR_dxi = (sum_dnumPRdx * U_PR_Den - U_PR_Num * sumdPAkdxi) / pow(U_PR_Den,2.);
                dPR_dyi = (sum_dnumPRdy * U_PR_Den - U_PR_Num * sumdPAkdyi) / pow(U_PR_Den,2.);
                // cout << "dpr dx dy = " << sum_dnumPRdx << " " << sum_dnumPRdy << " " << sumdPAkdxi << " " << sumdPAkdyi << " " << U_PR_Num << " " << U_PR_Den << " " << dPR_dxi << " " << dPR_dyi << endl;
                if (dPR_dxi!=dPR_dxi) {
                    cout    << "Warning, dPRdx issue " << sum_dnumPRdx << " " << U_PR_Den << " " << U_PR_Num << " " << sumdPAkdxi << endl;
                    outfile << "Warning, dPRdx issue " << sum_dnumPRdx << " " << U_PR_Den << " " << U_PR_Num << " " << sumdPAkdxi << endl;
#if defined(STANDALONE) || defined(UBUNTU)
                    datamutex.lock();
#endif
                    warnings5++; 
#if defined(STANDALONE) || defined(UBUNTU)
                    datamutex.unlock();
#endif
                }
                //if (dPR_dyi!=dPR_dyi) cout << "dPRdy issue " << sum_dnumPRdy << " " << U_PR_Den << " " << U_PR_Num << " " << sumdPAkdyi << endl;
            } else {
                cout    << "Warning, U_PR_Den is null " << endl;
                outfile << "Warning, U_PR_Den is null " << endl;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings5++; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
            }
        }

        // Now we compute the derivative of the ExposureFactor, which is needed for both U_GF and U_PS
        // -------------------------------------------------------------------------------------------

        // Accumulate the derivative of the utility with respect to x,y
        // Note, we use the measured Fg here, but we could opt for a saturated model instead
        // ---------------------------------------------------------------------------------
        // Note: as 
        //    U_GF = eta_gf * fs*invsigfs*EF, 
        // with 
        //    sqrt(1/rho) = EF, 
        // ie. 
        //    EF = sqrt(Rtot^2/Ntrials), 
        // we get the expression below. 
        //    U_GF = eta_gf * fs * invsigfs * Rtot/sqrt(Ntrials)
        // Note: if Ntrials increases, U_GF decreases as we are effectively integrating for more time to get the
        // same precision in flux. 
        // This is consistent with U_GF being proportional to the exposure, and the
        // exposure being inversely proportional to the number of trials per area (Ntrials/Rtot^2).
        // In particular the rhoinv factor under square root contributes as follows:
        //    dU/dx += U_GF /rhoinv^(1/2) * 1/2 rhoinv^(-1/2) * drhoinv/dx 
        //    drhoinv/dx = rhoinv*( 1/Rtot dRtot/dx + 1/Ntrials * dNtrials/dx).
        // dRtot/dx can be obtained by checking how R varies only for the detector at max r, indmaxR. (this is an 
        // approximation): Rtot = Rmax+Rslack = sqrt(x*x+y*y)+Rslack, dRtot/dx = x/Rmax (only for the outermost
        // detector!). This is computed before this loop.
        // dNtrials/dx is computed by looping on all positions of the trials, TryX0[is][iter], tryY0[is][iter].
        // dNactive/dx is computed above as sumdPAkdxi_noweight (but we do not use it anymore here)
        // ----------------------------------------------------------------------------------------------------

        // Compute dNtrials/dx,dy
        // ----------------------
        double dx = 0.25*DetectorSpacing; // Test displacement
        double dy = 0.25*DetectorSpacing; // Test displacement
        double dNx = 0.;
        double dNy = 0.;

        for (int is=Nevents; is<Nevents+Nbatch; is++) {  
  
            // NOTE: do we need to ask that these events are active to get these dnx, dny?

            // Check how many showers that were rejected would be accepted with dx,dy movement
            // -------------------------------------------------------------------------------
            for (int iter=0; iter<Ntrials[is]-1 && iter<maxIter; iter++) { // the Ntrials[is]-th was the successful trial, hence -1
                double r2newx = pow(xi+dx-TryX0[is][iter],2.) + pow(yi-TryY0[is][iter],2.);
                double r2newy = pow(xi-TryX0[is][iter],2.)    + pow(yi+dy-TryY0[is][iter],2.);        
                // Note that below we decrease Nx, Ny by one, not by Ntrials-iter, because we are testing all trials
                // -------------------------------------------------------------------------------------------------
                if (r2newx<Rslack2) dNx--; // A movement dx causes this failed trial to succeed, so Nx must decrease
                if (r2newy<Rslack2) dNy--; // Idem for dy, Ny 
                // Now check the other directions, properly accounting for the sign
                // ----------------------------------------------------------------
                r2newx = pow(xi-dx-TryX0[is][iter],2.) + pow(yi-TryY0[is][iter],2.);
                r2newy = pow(xi-TryX0[is][iter],2.)    + pow(yi-dy-TryY0[is][iter],2.);        
                if (r2newx<Rslack2) dNx++; // A negative dx caused a decrease in Ntrials, so dnx/dx will be positive
                if (r2newy<Rslack2) dNy++; // Idem for dy 
            }
            // Now check if the _accepted_ shower would fall out of accepted region just because of a move of 
            // the detector id we are checking
            // ----------------------------------------------------------------------------------------------
            bool pass = false;
            for (int jd=0; jd<Nunits && !pass; jd++) {
                if (jd==id) continue;
                double r2 = pow(x[jd]-TrueX0[is],2.) + pow(y[jd]-TrueY0[is],2.);
                if (r2<Rslack2) pass = true; // Some other detector is anyway closer than Rslack
            }
            if (!pass) { // That shower was accepted in Nbatch thanks to being close to id only
                         // So we check if a movement of id would void that
                double r2newx = pow(xi+dx-TrueX0[is],2.) + pow(yi-TrueY0[is],2.);
                double r2newy = pow(xi-TrueX0[is],2.)    + pow(yi+dy-TrueY0[is],2.);        
                // If the movement caused the accepted shower to get rejected, the number of trials
                // would then have to double, as the best estimate we have for the accept prob is 1/ntrials.
                // -----------------------------------------------------------------------------------------
                if (r2newx>Rslack2) dNx += Ntrials[is]; 
                if (r2newy>Rslack2) dNy += Ntrials[is]; 
                // Also check the other direction, with opposite sign in the effect
                // ----------------------------------------------------------------
                r2newx = pow(xi-dx-TrueX0[is],2.) + pow(yi-TrueY0[is],2.);
                r2newy = pow(xi-TrueX0[is],2.)    + pow(yi-dy-TrueY0[is],2.);        
                if (r2newx>Rslack2) dNx -= Ntrials[is]; 
                if (r2newy>Rslack2) dNy -= Ntrials[is]; 
            }
        } // End loop on Nbatch events
        double dNtrials_dx = dNx/dx;
        double dNtrials_dy = dNy/dy;
        // Now we can compute the total effect on ExposureFactor of this detector's movement
        // ---------------------------------------------------------------------------------
        dExpFac_dx = dTotalRspan_dx/sqrt(totNtrials) - 0.5*TotalRspan*pow(totNtrials,-1.5)*dNtrials_dx;
        dExpFac_dy = dTotalRspan_dy/sqrt(totNtrials) - 0.5*TotalRspan*pow(totNtrials,-1.5)*dNtrials_dy;

        double dU_dxiPS = 0.;
        double dU_dyiPS = 0.;
        if (PeVSource) {
            double sum_dens2_2 = pow(sum_dens2,2.);
            // Those below are derivatives of sigma_theta^2 and sigma_phi^2, which are defined as <x^2> - <x>^2 with x = theta_m-theta_t (or phi).
            // We write them as (dnum1_dx*den1 - num1*dden1_dx)/den1^2 -2(num2/den2)*(dnum2_dx*den2-num2*dden2_dx)/den2^2
            // -----------------------------------------------------------------------------------------------------------------------------------
            double sigma2t = sum_num1s2t/sum_dens2 - pow(sum_num2s2t/sum_dens2,2.);
            double sigma2p = sum_num1s2p/sum_dens2 - pow(sum_num2s2p/sum_dens2,2.);
            if (sigma2t<=0.) {
                cout    << "Warning, sigma2t was negative. fixed it to epsilon. " << endl;
                outfile << "Warning, sigma2t was negative. fixed it to epsilon. " << endl;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings7++; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
                sigma2t = epsilon; // While waiting to debug the above expressions, we safeguard them
            }
            if (sigma2p<=0.) {
                cout    << "Warning, sigma2p was negative. fixed it to epsilon. " << endl;
                outfile << "Warning, sigma2p was negative. fixed it to epsilon. " << endl;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings7++; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
                sigma2p = epsilon;
            }
            double ds2t_dx = (sum_dnum1s2t_dx*sum_dens2 - sum_num1s2t*sumdPAkdxi)/sum_dens2_2
                             -2.*(sum_num2s2t/sum_dens2)*(sum_dnum2s2t_dx*sum_dens2-sum_num2s2t*sumdPAkdxi)/sum_dens2_2;
            double ds2t_dy = (sum_dnum1s2t_dy*sum_dens2 - sum_num1s2t*sumdPAkdyi)/sum_dens2_2 
                             -2.*(sum_num2s2t/sum_dens2)*(sum_dnum2s2t_dy*sum_dens2-sum_num2s2t*sumdPAkdyi)/sum_dens2_2;
            double ds2p_dx = (sum_dnum1s2p_dx*sum_dens2 - sum_num1s2p*sumdPAkdxi)/sum_dens2_2 
                             -2.*(sum_num2s2p/sum_dens2)*(sum_dnum2s2p_dx*sum_dens2-sum_num2s2p*sumdPAkdxi)/sum_dens2_2;
            double ds2p_dy = (sum_dnum1s2p_dy*sum_dens2 - sum_num1s2p*sumdPAkdyi)/sum_dens2_2 
                             -2.*(sum_num2s2p/sum_dens2)*(sum_dnum2s2p_dy*sum_dens2-sum_num2s2p*sumdPAkdyi)/sum_dens2_2;
            // The calculation for SidebandArea is:
            // SBA = pow(20*1.4,2)*sqrt[sigma2t*sigma2p]
            // dSBA/dxi = pow(20*1.4,2)/(2*sqrt[])*(ds2t/dx*s2p + ds2p/dx*s2t)
            // ---------------------------------------------------------------
            double commonfactor = pow(20.*1.4,2.)/(2.*sqrt(sigma2t*sigma2p));
            double dAsb_dxj = commonfactor*(ds2t_dx*sigma2p+ds2p_dx*sigma2t);
            double dAsb_dyj = commonfactor*(ds2t_dy*sigma2p+ds2p_dy*sigma2t);
            // if (dAsb_dxj!=dAsb_dxj) cout << "Warning, sigma2t, sigma2p = " << sigma2t << " " << sigma2p << " ds term = " << ds2t_dx + ds2p_dx << endl;
            // if (dAsb_dyj!=dAsb_dyj) cout << "Warning, sigma2t, sigma2p = " << sigma2t << " " << sigma2p << " ds term = " << ds2t_dy + ds2p_dy << endl;

            // As U = 1/N_5sigma = k * EF * 4 / (25+20sqrt(B)), we get
            //     dU/dx = U * [1/EF * dEF/dx - 1/(25+20sqrt(B) * d(25+20sqrt(B)/dB * dB/dx]
            // with
            //     d(25+20sqrt(B))/dB = 10/sqrt(B)
            // and
            //     dB/dx = B * [1/EF * dEF/dx + 1/SBA * dSBA/dx + 1/EIW * dEIW/dx - 1/SWE * dSWE/dx]
            // -------------------------------------------------------------------------------------
            double dEfr_dxj = dEfrdxj_factor1 + dEfrdxj_factor2;
            double dEfr_dyj = dEfrdyj_factor1 + dEfrdyj_factor2;
            if (useN5s) {
                dU_dxiPS = U_PS * (dExpFac_dx/ExposureFactor -1./(25.+20.*sqrt(PS_B))*10./sqrt(PS_B)*
                           PS_B * (dExpFac_dx/ExposureFactor + dAsb_dxj/PS_SidebandArea + dEfr_dxj/PS_EintervalWeight - PS_sumdPAkdxi/PS_SumWeightsE));
                dU_dyiPS = U_PS * (dExpFac_dy/ExposureFactor -1./(25.+20.*sqrt(PS_B))*10./sqrt(PS_B)*
                           PS_B * (dExpFac_dy/ExposureFactor + dAsb_dyj/PS_SidebandArea + dEfr_dyj/PS_EintervalWeight - PS_sumdPAkdyi/PS_SumWeightsE));
                if (dU_dxiPS!=dU_dxiPS || dU_dyiPS!=dU_dyiPS) {
                    cout    << "Warning, dups problem: B= " << PS_B << " dAsbdx,dy = " << dAsb_dxj << " " << dAsb_dyj << " dEfrdx = " << dEfr_dxj << " " << dEfr_dyj << " EIW = " << PS_EintervalWeight << " sdPdx,dy = " << PS_sumdPAkdxi << " " << PS_sumdPAkdyi << " SWE = " << PS_SumWeightsE << endl;  
                    outfile << "Warning, dups problem: B= " << PS_B << " dAsbdx,dy = " << dAsb_dxj << " " << dAsb_dyj << " dEfrdx = " << dEfr_dxj << " " << dEfr_dyj << " EIW = " << PS_EintervalWeight << " sdPdx,dy = " << PS_sumdPAkdxi << " " << PS_sumdPAkdyi << " SWE = " << PS_SumWeightsE << endl;  
#if defined(STANDALONE) || defined(UBUNTU)
                    datamutex.lock();
#endif
                    warnings5++; 
#if defined(STANDALONE) || defined(UBUNTU)
                    datamutex.unlock();
#endif
                }
            } else {
                // NNBB to be retested
                // 
                // Get derivatives of the sum of weights of events in this batch, for this detector position
                // -----------------------------------------------------------------------------------------
                // The calculation below goes as follows:
                // SB/B = sqrt(sum w^2/ (sum w)^2 + (sigma_Fs/Fs)^2)
                // d(SB/B)/dx = 1/(2*SB/B) * (1/sum(w)^2*d(sum w^2)/dx +sum(w^2)*d((sum w)^-2/dx) + 
                //              + d/dx(1/Fs^2 * (1/sigma_Fs)^-2) )
                // We leverage having computed PS_sumdPAkdxi, dxiw as sums above; the latter is half
                // of the derivative  d(sum w^2) /dx. We also use the inverse sigma on Fs computed earlier.
                // ------------------------------------------------------------------------------------------------------
                PS_dBdx = PS_B * (dAsb_dxj/PS_SidebandArea + dEfr_dxj/PS_EintervalWeight - PS_sumdPAkdxi/PS_SumWeightsE);
                PS_dsigmaBoverB_dx = 1./(2.*PS_sigmaBoverB)*(2.*PS_sumdPAkdxiw/(pow(PS_SumWeightsE,2.)) + 
                                                             -2.*PS_SumWeightsE2*pow(PS_SumWeightsE,-3.)*PS_sumdPAkdxi +
                                                             pow(MeasFg,-2.)*(-2.*pow(inv_sigmafs,-3.)*d_invsigfs_dx));
                PS_dBdy = PS_B * (dAsb_dyj/PS_SidebandArea + dEfr_dyj/PS_EintervalWeight - PS_sumdPAkdyi/PS_SumWeightsE);
                PS_dsigmaBoverB_dy = 1./(2.*PS_sigmaBoverB)*(2.*PS_sumdPAkdyiw/(pow(PS_SumWeightsE,2.)) + 
                                                             -2.*PS_SumWeightsE2*pow(PS_SumWeightsE,-3.)*PS_sumdPAkdyi +
                                                             pow(MeasFg,-2.)*(-2.*pow(inv_sigmafs,-3.)*d_invsigfs_dy));
                // cout << " dsps x: dAsb = " << dAsb_dxj << " sumdpxiw = " << PS_sumdPAkdxiw << " sumdpakxiw" << PS_sumdPAkdxi << endl; // " PS_sigmaBoverB = " << PS_sigmaBoverB << " PS_SumWeightsE = " << PS_SumWeightsE << " inv_sigmafs = " << inv_sigmafs << endl;
                // cout << " dsps y: dAsb = " << dAsb_dyj << " sumdpyiw = " << PS_sumdPAkdyiw << " sumdpakyiw" << PS_sumdPAkdyi << endl;
                double N3S = N_3S();
                if (N3S!=N3S || N3S==0) {
                    N3S = 1.; // kludge 
                    cout    << "Warning N3S kludge" << endl;
                    outfile << "Warning N3S kludge" << endl;
#if defined(STANDALONE) || defined(UBUNTU)
                    datamutex.lock();
#endif
                    warnings6++; 
#if defined(STANDALONE) || defined(UBUNTU)
                    datamutex.unlock();
#endif
                }
                dU_dxiPS = -U_PS/coeff_PS*N_3S(id,1)/N3S;
                dU_dyiPS = -U_PS/coeff_PS*N_3S(id,2)/N3S;
                if (dU_dxiPS!=dU_dxiPS) {
                    cout    << "Warning dups: U_PS = " << U_PS << " coef = " << coeff_PS 
                            << " " << dAsb_dxj << " " << PS_sumdPAkdxi 
                            << " N_3s(" << id << ",1)=" << N_3S(id,1) << " N_3S=" << N3S << endl;
                    outfile << "Warning dups: U_PS = " << U_PS << " coef = " << coeff_PS 
                            << " " << dAsb_dxj << " " << PS_sumdPAkdxi 
                            << " N_3s(" << id << ",1)=" << N_3S(id,1) << " N_3S=" << N3S << endl;
#if defined(STANDALONE) || defined(UBUNTU)
                    datamutex.lock();
#endif
                    warnings3++; 
#if defined(STANDALONE) || defined(UBUNTU)
                    datamutex.unlock();
#endif
                    TerminateAbnormally();
                }
            }
        }

        // Now we can compute the utility variation with dx, dy of this detector
        // ---------------------------------------------------------------------
        if (MeasFg!=0. && inv_sigmafs!=0. && N_active!=0. && totNtrials>0) { 
            double dU_dxiGF = eta_GF * U_GF /coeff_GF * (d_fs_dx/MeasFg + d_invsigfs_dx/inv_sigmafs + dExpFac_dx/ExposureFactor);
                                // NB all the factors below account for utility variations due to flux variations due to varying exposure.
                                // - 0.5*dNtrials_dx/totNtrials // this comes from ExposureFactor being prop to (totNtrials^-1/2), so 1/(totNtrials^-1/2)*(-1/2 totNtrials^-3/2)*dtotNtrials/dx
                                // + dTotalRspan_dx/TotalRspan  // this is the other part of ExposureFactor, which is prop to TotalRspan
                                // + dRtot_dx/TotalRspan // the above includes this
                                // - 0.5*sumdPAkdxi_noweight/N_active ---> this is present only if ExposureFactor has Nbatch/Nactive, but we removed it.
                                //);
            //cout << "dx " << dTotalRspan_dx/TotalRspan << " " << dRtot_dx/TotalRspan << endl;
            double dU_dxiIR = eta_IR*dIR_dxi;
            double dU_dxiPR = eta_PR*dPR_dxi;
            double dU_dxiTC = 0.;
            if (UseAreaCost) {
                ComputeUtilityArea (id,1); // Defines dUA_dx for this detector;
                dU_dxiTC += eta_TA*dUA_dx;
            }
            if (UseLengthCost) {
                ComputeUtilityLength(id,1);
                dU_dxiTC += eta_TL*dUL_dx;
            }            
            if (PeVSource) {
                dU_dxi[id] = eta_PS*dU_dxiPS + dU_dxiTC;
            } else {
                dU_dxi[id] = dU_dxiGF + dU_dxiIR + dU_dxiPR + dU_dxiTC;
            }            
            if (dU_dxi[id]!=dU_dxi[id]) {
                cout    << "Trouble with dUdx: id= " << id << " dfs/dx=" << d_fs_dx << " fg=" << MeasFg << " d1/sdx=" << d_invsigfs_dx << " 1/s=" << inv_sigmafs 
                        << " dntdx=" << dNtrials_dx << " drdx=" << dRtot_dx << " sp=" << (TotalRspan) << " dprdx=" << dPR_dxi 
                        << " dudxps=" << dU_dxiPS
                        << " dudxgf=" << dU_dxiGF << " dudxir=" << dU_dxiIR << " dudxpr=" << dU_dxiPR << "dudxtc=" << dU_dxiTC << endl;
                outfile << "Trouble with dUdx: id= " << id << " dfs/dx=" << d_fs_dx << " fg=" << MeasFg << " d1/sdx=" << d_invsigfs_dx << " 1/s=" << inv_sigmafs 
                        << " dntdx=" << dNtrials_dx << " drdx=" << dRtot_dx << " sp=" << (TotalRspan) << " dprdx=" << dPR_dxi 
                        << " dudxps=" << dU_dxiPS
                        << " dudxgf=" << dU_dxiGF << " dudxir=" << dU_dxiIR << " dudxpr=" << dU_dxiPR << " dudxtc=" << dU_dxiTC << endl;
                double c = ConvexHull(); 
                cout    << "Convex hull area is " << c << endl;
                outfile << "Convex hull area is " << c << endl;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings1++; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
            }
            double dU_dyiGF = eta_GF * U_GF /coeff_GF * (d_fs_dy/MeasFg + d_invsigfs_dy/inv_sigmafs + dExpFac_dy/ExposureFactor);
                              // - 0.5*dNtrials_dy/totNtrials  
                              // + dTotalRspan_dy/TotalRspan
                              // + dRtot_dy/TotalRspan // the above term includes this 
                              // - 0.5*sumdPAkdyi_noweight/N_active // -> this not present if we remove Nbatch/Nactive in ExposureFactor
                              // );
            //cout << "dy " << dTotalRspan_dy/TotalRspan << " " << dRtot_dy/TotalRspan << endl;
            double dU_dyiIR = eta_IR*dIR_dyi;
            double dU_dyiPR = eta_PR*dPR_dyi;
    	    double dU_dyiTC = 0.;
            if (UseAreaCost) {
                ComputeUtilityArea (id,2); // Defines dUA_dy for this detector;
                dU_dyiTC += eta_TA*dUA_dy;
            }
            if (UseLengthCost) {
                ComputeUtilityLength(id,2);
                dU_dyiTC += eta_TL*dUL_dy;
            }            
            if (PeVSource) {
                dU_dyi[id] = eta_PS*dU_dyiPS + dU_dyiTC;
            } else {
                dU_dyi[id] = dU_dyiGF + dU_dyiIR + dU_dyiPR + dU_dyiTC;
            }
            if (dU_dyi[id]!=dU_dyi[id]) {
                cout    << "Trouble with dUdy: " << d_fs_dy << " " << MeasFg << " " << d_invsigfs_dy << " " << inv_sigmafs 
                        << " " << dNtrials_dy << " " << dRtot_dy << " " << (TotalRspan) << " " << dPR_dyi 
                        << " dudyps=" << dU_dyiPS
                        << dU_dyiGF << " " << dU_dyiIR << " " << dU_dyiPR << " " << dU_dyiTC << endl;
                outfile << "Trouble with dUdy: " << d_fs_dy << " " << MeasFg << " " << d_invsigfs_dy << " " << inv_sigmafs 
                        << " " << dNtrials_dy << " " << dRtot_dy << " " << (TotalRspan) << " " << dPR_dyi 
                        << " dudyps=" << dU_dyiPS
                        << dU_dyiGF << " " << dU_dyiIR << " " << dU_dyiPR << " " << dU_dyiTC << endl;
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                warnings1++; 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
            }   

            // Compute size of dudx and dudy
            // -----------------------------
            /*
             if (dU_dxiGF!=0. && dU_dyiGF!=0.) {
                 double tmp = fabs(dU_dxiGF)/fabs(dU_dyiGF);
                 if (tmp>10.) tmp = 10.;
                 if (tmp<0.1) tmp = 0.1;
                 sumratio_dudxdy_GF += log(tmp);
            }
            if (dU_dxiIR!=0. && dU_dyiIR!=0.) {
                 double tmp = fabs(dU_dxiIR)/fabs(dU_dyiIR);
                 if (tmp>10.) tmp = 10.;
                 if (tmp<0.1) tmp = 0.1;
                 sumratio_dudxdy_IR += log(tmp);
            }
            if (dU_dxiPR!=0. && dU_dyiPR!=0.) {
                double tmp = fabs(dU_dxiPR)/fabs(dU_dyiPR);
                if (tmp>10.) tmp = 10.;
                if (tmp<0.1) tmp = 0.1;
                sumratio_dudxdy_PR += log(tmp);
            }
            */

            // Sanity checks
            // -------------
            //if (fabs(dU_dxi[id])>100. || fabs(dU_dyi[id]>100.)) {
            //    cout << "     Large dUdx. Components: " << dU_dxiGF << " " << dU_dxiIR << " " << dU_dxiPR << " " << dU_dxiPS;
            //    float r2min = largenumber;
            //    for (int is = Nevents; is < Nevents+Nbatch; is++) {
            //        float r2 = pow(TrueX0[is]-x[id],2.)+pow(TrueY0[is]-y[id],2.);
            //        if (r2<r2min) r2min = r2;
            //    }
            //    cout << " min r = " << sqrt(r2min) << " invsig = " << inv_sigmafs << endl;
            //}

            // Compare dudx, dudy for ps and cost
            // ----------------------------------
            // sumduc += sqrt(pow(dU_dxiTC,2.)+pow(dU_dyiTC,2.));
            // sumdup += eta_PS*sqrt(pow(dU_dxiPS,2.)+pow(dU_dyiPS,2.));
            // cout << " sumdup = " << sumdup << " dupsdx,dy = " << dU_dxiPS << " " << dU_dyiPS << endl;

            
            // Keep track of average log(dudx) for components, to rescale learning rates dynamically
            // -------------------------------------------------------------------------------------
            double dU_drGF = pow(dU_dxiGF*dU_dxiGF+dU_dyiGF*dU_dyiGF,0.5); // this is not really a dU/dr, but we still want the modulus of the 2D gradient
            double dU_drIR = pow(dU_dxiIR*dU_dxiIR+dU_dyiIR*dU_dyiIR,0.5);
            double dU_drPR = pow(dU_dxiPR*dU_dxiPR+dU_dyiPR*dU_dyiPR,0.5);
            double dU_drTC = pow(dU_dxiTC*dU_dxiTC+dU_dyiTC*dU_dyiTC,0.5);
            if (DynamicLR) {
                ave_dUgf[id] = log(epsilon+dU_drGF);
                ave_dUir[id] = log(epsilon+dU_drIR);
                ave_dUpr[id] = log(epsilon+dU_drPR);
                ave_dUtc[id] = log(epsilon+dU_drTC);
            }
            if (PlotThis[7]) { 
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.lock();
#endif
                DUGF->Fill(log(epsilon+dU_drGF)); 
                DUIR->Fill(log(epsilon+dU_drIR));
                DUPR->Fill(log(epsilon+dU_drPR));
                DUTC->Fill(log(epsilon+dU_drTC));
#if defined(STANDALONE) || defined(UBUNTU)
                datamutex.unlock();
#endif
            }
            // cout << "id = " << id << " dudx = " << dU_dxi[id] << " dn/n = " << dNtrials_dx/totNtrials 
            //      << " dR/R = " << dRtot_dx/(TotalRspan) << " dPa/Pa = " <<  sumdPAkdxi_noweight/N_active << endl;
            // cout << "id = " << id << " dudy = " << dU_dyi[id] << " dn/n = " << dNtrials_dy/totNtrials 
            //      << " dR/R = " << dRtot_dy/(TotalRspan) << " dPa/Pa = " <<  sumdPAkdyi_noweight/N_active << endl;
            // Fill histograms of relative motion for three utility components
            // ---------------------------------------------------------------
            if (PlotThis[13] || PlotThis[14]) {
                double modGF = sqrt(dU_dxiGF*dU_dxiGF+dU_dyiGF*dU_dyiGF);
                double modIR = sqrt(dU_dxiIR*dU_dxiIR+dU_dyiIR*dU_dyiIR);
                double modPR = sqrt(dU_dxiPR*dU_dxiPR+dU_dyiPR*dU_dyiPR);
                if (PlotThis[13]) {
                    double ctgi = (dU_dxiGF*dU_dxiIR+dU_dyiGF*dU_dyiIR)/(modGF*modIR);
                    double ctgp = (dU_dxiGF*dU_dxiPR+dU_dyiGF*dU_dyiPR)/(modGF*modPR);
#if defined(STANDALONE) || defined(UBUNTU)
                    datamutex.lock();
#endif
                    ThGIvsThGP->Fill(acos(ctgp),acos(ctgi)); // lock the mutex?
#if defined(STANDALONE) || defined(UBUNTU)
                    datamutex.unlock();
#endif
                } 
                if (PlotThis[14]) {
                    double maxGI = modGF;
                    if (maxGI<modIR) maxGI = modIR;
                    double maxGP = modGF;
                    if (maxGP<modPR) maxGP = modPR;
#if defined(STANDALONE) || defined(UBUNTU)
                    datamutex.lock();
#endif
                    LrGIvsLrGP->Fill(acos((modGF-modPR)/maxGP),acos((modGF-modIR)/maxGI)); 
#if defined(STANDALONE) || defined(UBUNTU)
                    datamutex.unlock();
#endif
                }
            }
        } else {
            cout    << "Warning, ivision by zero in dUdx calculation. " << endl;
            outfile << "Warning, division by zero in dUdx calculation. " << endl;
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.lock();
#endif
            warnings2++; 
#if defined(STANDALONE) || defined(UBUNTU)
            datamutex.unlock();
#endif
            TerminateAbnormally();
        }
#ifdef PLOTS
        //dUdx->Fill(1.*epoch,fabs(dU_dxi[id]));
        //dUdx->Fill(1.*epoch,fabs(dU_dyi[id])); // One histogram for both coordinates
#endif

        // Check if we need to add repulsive gradient from forbidden region
        // NOTE: These calls to ForbiddenRegion do not involve writing on arrays
        // that may be shared by different threads, because we are calling them
        // with mode=1 and thus only idstar coordinates are modified.
        // ---------------------------------------------------------------------
        if (VoidRegion) {
            //IncludeOnlyFR = false;
            //FRpar[0] = 0.5;
            //FRpar[1] = 200.;
            //ForbiddenRegion (1,id,1); // in y region below line y=0.5x+200
            //IncludeOnlyFR = false;
            //FRpar[0] = -100.;
            //FRpar[1] = 300.;
            //ForbiddenRegion (2,id,1); // in x region [-100,300] 
            //IncludeOnlyFR = true;
            //FRpar[0] = 0.;
            //FRpar[1] = 0.;
            //FRpar[2] = 600.;
            //ForbiddenRegion(0,id,1); // within a 600 m circle
            // Pampalabola: triangle at positions
            // A = (-3499.,2000.) B = (1800,2000) C = (1800,-3508). BAC is -0.804735, intercept -816
            // ---------------------
            IncludeOnlyFR = false;
            FRpar[0] = -3499.;
            FRpar[1] = 1800.;
            ForbiddenRegion(2,id,1); // Vertical semiplane
            dU_dxi[id] += dU_dxiFR;
            dU_dyi[id] += dU_dyiFR;

            IncludeOnlyFR = true;
            FRpar[0] = -tan(0.804735);
            FRpar[1] = -1637.;
            ForbiddenRegion(1,id,1); // Non vertical semiplane defined by y=mx+q boundary
            dU_dxi[id] += dU_dxiFR;
            dU_dyi[id] += dU_dyiFR;

            IncludeOnlyFR = false;
            FRpar[0] = 0.;
            FRpar[1] = 2000.;
            ForbiddenRegion(1,id,1); // Non vertical semiplane defined by y=mx+q boundary
            dU_dxi[id] += dU_dxiFR;
            dU_dyi[id] += dU_dyiFR;
        }

    } // End id loop on dets

    /*
    cout << endl;
    cout << "     Avg log ratio of gradient strengths x/y:";
    cout << " GF = " << sumratio_dudxdy_GF;
    cout << " IR = " << sumratio_dudxdy_IR;
    cout << " PR = " << sumratio_dudxdy_PR;
    */

    // Record end time
    // ---------------
    //std::clock_t ending_time = std::clock();    
    // Calculate the duration of the loop
    // ----------------------------------
    //double time_duration = double(ending_time - starting_time)/CLOCKS_PER_SEC;
    // Output the duration in seconds
    // ------------------------------
    //std::cout << "Time taken by function: " << time_duration << endl; 

    // End of routine, back to caller
    // ------------------------------
    return;

}


// Function that fills parameters of showers
// -----------------------------------------
int ReadShowers () {

#ifdef STANDALONE
    string trainPath  = GlobalPath + "Model/"; // "/lustre/cmswork/dorigo/swgo/MT/Model/";
#endif
#ifdef UBUNTU
    string trainPath  = GlobalPath + "Model/"; // "/home/tommaso/Work/swgo/MT/Model/";
#endif
#ifdef INROOT
    string trainPath  = "./SWGO/Model/";
#endif
    ifstream asciifile_g, asciifile_p;

    // Read gamma parameters
    // 
    // The model is 
    //      dn(t,e)/dr = p0(t,e)*exp(-p1(t,e)*r^p2(t,e))
    // with
    //      p0(t,e) = q00(t)*exp(q10(t)*ln(e)^q20(t))
    //      p1(t,e) = q10(t)*exp(q11(t)*ln(e)^q21(t))
    //      p2(t,e) = q20(t)*exp(q12(t)*ln(e)^q22(t))
    // Below, we read sequentially the parameters q from a photon file, and then from a proton file.
    // The sequence of parameters in each file is as follows: we start with electron secondaries and have:
    //      q00(t=0), q01(t=0), q02(t=0)
    //      q10(t=0), q11(t=0), q12(t=0)
    //      q20(t=0), q21(t=0), q22(t=0)
    // then
    //      q00(t=1), q01(t=1), q02(t=1)
    //      q10(t=1), q11(t=1), q12(t=1)
    //      q20(t=1), q21(t=1), q22(t=1)
    // then t=2 and t=3. 
    // Then the same structure is repeated for muons.    
    //
    // The nomenclature of the variables that store these parameters is: for electrons in gamma showers,
    //      for t=0, PXeg1_p[j][i] = q_ji(0)
    //      for t=1, PXeg2_p[j][i] = q_ji(1)
    // and so on. For muons in gamma showers the name is PXmg. For electrons in proton showers, PXep, and
    // for muons in proton showers, PXmp.
    //
    // For the sake of merging of q1 and p1, which were fit in the surrogate model but can be gotten rid of
    // (as a*exp(b*R^c) = a'*exp(R^c) with a' == a*exp(b) ),
    // we need to compute, e.g. for electrons in gamma showers:
    //     q0'(t=0) =  PXeg1_p[0][0] * exp (PXeg1_p[1][0])
    // ----------------------------------------------------------------------------------------------------
    std::stringstream sstr_g;
    sstr_g << "Fit_Photon_10_pars";
    string traininglist_g = trainPath  + sstr_g.str() + ".txt";
    asciifile_g.open(traininglist_g);
    double e;
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_g >> e;
            PXeg1_p[jp][ip] = e;
        }
        if (PXeg1_p[jp][0]*PXeg1_p[jp][1]*PXeg1_p[jp][2]==0) {
            cout    << "Warning, p" << jp << "eg1 = " << PXeg1_p[jp][0] << " " << PXeg1_p[jp][1] << " " << PXeg1_p[jp][2] << endl;
            outfile << "Warning, p" << jp << "eg1 = " << PXeg1_p[jp][0] << " " << PXeg1_p[jp][1] << " " << PXeg1_p[jp][2] << endl;
            warnings7++;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_g >> e;
            PXeg2_p[jp][ip] = e;
        }
        if (PXeg2_p[jp][0]*PXeg2_p[jp][1]*PXeg2_p[jp][2]==0) {
            cout    << "Warning, p" << jp << "eg2 = " << PXeg2_p[jp][0] << " " << PXeg2_p[jp][1] << " " << PXeg2_p[jp][2] << endl;
            outfile << "Warning, p" << jp << "eg2 = " << PXeg2_p[jp][0] << " " << PXeg2_p[jp][1] << " " << PXeg2_p[jp][2] << endl;
            warnings7++;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_g >> e;
            PXeg3_p[jp][ip] = e;
        }
        if (PXeg3_p[jp][0]*PXeg3_p[jp][1]*PXeg3_p[jp][2]==0) {
            cout    << "Warning, p" << jp << "eg3 = " << PXeg3_p[jp][0] << " " << PXeg3_p[jp][1] << " " << PXeg3_p[jp][2] << endl;
            outfile << "Warning, p" << jp << "eg3 = " << PXeg3_p[jp][0] << " " << PXeg3_p[jp][1] << " " << PXeg3_p[jp][2] << endl;
            warnings7++;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_g >> e;
            PXeg4_p[jp][ip] = e;
        }
        if (PXeg4_p[jp][0]*PXeg4_p[jp][1]*PXeg4_p[jp][2]==0) {
            cout    << "Warning, p" << jp << "eg4 = " << PXeg4_p[jp][0] << " " << PXeg4_p[jp][1] << " " << PXeg4_p[jp][2] << endl;
            outfile << "Warning, p" << jp << "eg4 = " << PXeg4_p[jp][0] << " " << PXeg4_p[jp][1] << " " << PXeg4_p[jp][2] << endl;
            warnings7++;
            return 1;
        }
    }
    // Tmp: for the sake of producing a table of these parameters in the paper, we may merge q0 and q1, q0' = q0 exp(q1)
    // and the result can be reported in a table as in the appendix, as follows:
    // -----------------------------------------------------------------------------------------------------------------
    /*
    for (int ip=0; ip<3; ip++) {
        cout << "% " << PXeg1_p[0][ip]*exp(PXeg1_p[1][ip]) << " % " << PXeg1_p[2][ip] " \\" << endl;
        cout << "% " << PXeg2_p[0][ip]*exp(PXeg2_p[1][ip]) << " % " << PXeg2_p[2][ip] " \\" << endl;
        cout << "% " << PXeg3_p[0][ip]*exp(PXeg3_p[1][ip]) << " % " << PXeg3_p[2][ip] " \\" << endl;
        cout << "% " << PXeg4_p[0][ip]*exp(PXeg4_p[1][ip]) << " % " << PXeg4_p[2][ip] " \\" << endl;
    }
    */


    // Done with electrons from gammas, now pars of muons from gammas
    // --------------------------------------------------------------
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_g >> e;
            PXmg1_p[jp][ip] = e;
        }
        if (PXmg1_p[jp][0]*PXmg1_p[jp][1]*PXmg1_p[jp][2]==0 && jp!=1) { // par1 is 0 for muons
            cout    << "Warning, p" << jp << "mg1 = " << PXmg1_p[jp][0] << " " << PXmg1_p[jp][1] << " " << PXmg1_p[jp][2] << endl;
            outfile << "Warning, p" << jp << "mg1 = " << PXmg1_p[jp][0] << " " << PXmg1_p[jp][1] << " " << PXmg1_p[jp][2] << endl;
            warnings7++;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_g >> e;
            PXmg2_p[jp][ip] = e;
        }
        if (PXmg2_p[jp][0]*PXmg2_p[jp][1]*PXmg2_p[jp][2]==0 && jp!=1) { // par1 is 0 for muons
            cout    << "Warning, p" << jp << "mg2 = " << PXmg2_p[jp][0] << " " << PXmg2_p[jp][1] << " " << PXmg2_p[jp][2] << endl;
            outfile << "Warning, p" << jp << "mg2 = " << PXmg2_p[jp][0] << " " << PXmg2_p[jp][1] << " " << PXmg2_p[jp][2] << endl;
            warnings7++;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_g >> e;
            PXmg3_p[jp][ip] = e;
        }
        if (PXmg3_p[jp][0]*PXmg3_p[jp][1]*PXmg3_p[jp][2]==0 && jp!=1) { // par1 is 0 for muons
            cout    << "Warning, p" << jp << "mg3 = " << PXmg3_p[jp][0] << " " << PXmg3_p[jp][1] << " " << PXmg3_p[jp][2] << endl;
            outfile << "Warning, p" << jp << "mg3 = " << PXmg3_p[jp][0] << " " << PXmg3_p[jp][1] << " " << PXmg3_p[jp][2] << endl;
            warnings7++;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_g >> e;
            PXmg4_p[jp][ip] = e;
        }
        if (PXmg4_p[jp][0]*PXmg4_p[jp][1]*PXmg4_p[jp][2]==0 && jp!=1) { // par1 is 0 for muons
            cout    << "Warning, p" << jp << "mg4 = " << PXmg4_p[jp][0] << " " << PXmg4_p[jp][1] << " " << PXmg4_p[jp][2] << endl;
            outfile << "Warning, p" << jp << "mg4 = " << PXmg4_p[jp][0] << " " << PXmg4_p[jp][1] << " " << PXmg4_p[jp][2] << endl;
            warnings7++;
            return 1;
        }
    }
    asciifile_g.close();

    // Proton data now
    // ---------------
    std::stringstream sstr_p;
    sstr_p << "Fit_Proton_2_pars";
    string traininglist_p = trainPath  + sstr_p.str() + ".txt";
    asciifile_p.open(traininglist_p);
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_p >> e;
            PXep1_p[jp][ip] = e;
        }
        if (PXep1_p[jp][0]*PXep1_p[jp][1]*PXep1_p[jp][2]==0) {
            cout    << "Warning in readshowers, p" << jp << "ep1 = " << PXep1_p[jp][0] << " " << PXep1_p[jp][1] << " " << PXep1_p[jp][2] << endl;
            outfile << "Warning in readshowers, p" << jp << "ep1 = " << PXep1_p[jp][0] << " " << PXep1_p[jp][1] << " " << PXep1_p[jp][2] << endl;
            warnings1++;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_p >> e;
            PXep2_p[jp][ip] = e;
        }
        if (PXep2_p[jp][0]*PXep2_p[jp][1]*PXep2_p[jp][2]==0) {
            cout    << "Warning, p" << jp << "ep2 = " << PXep2_p[jp][0] << " " << PXep2_p[jp][1] << " " << PXep2_p[jp][2] << endl;
            outfile << "Warning in readshowers, p" << jp << "ep2 = " << PXep2_p[jp][0] << " " << PXep2_p[jp][1] << " " << PXep2_p[jp][2] << endl;
            warnings1++;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_p >> e;
            PXep3_p[jp][ip] = e;
        }
        if (PXep3_p[jp][0]*PXep3_p[jp][1]*PXep3_p[jp][2]==0) {
            cout    << "Warning in readshowers, p" << jp << "ep3 = " << PXep3_p[jp][0] << " " << PXep3_p[jp][1] << " " << PXep3_p[jp][2] << endl;
            outfile << "Warning in readshowers, p" << jp << "ep3 = " << PXep3_p[jp][0] << " " << PXep3_p[jp][1] << " " << PXep3_p[jp][2] << endl;
            warnings1++;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_p >> e;
            PXep4_p[jp][ip] = e;
        }
        if (PXep4_p[jp][0]*PXep4_p[jp][1]*PXep4_p[jp][2]==0) {
            cout    << "Warning in readshowers, p" << jp << "ep4 = " << PXep4_p[jp][0] << " " << PXep4_p[jp][1] << " " << PXep4_p[jp][2] << endl;
            outfile << "Warning in readshowers, p" << jp << "ep4 = " << PXep4_p[jp][0] << " " << PXep4_p[jp][1] << " " << PXep4_p[jp][2] << endl;
            warnings1++;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_p >> e;
            PXmp1_p[jp][ip] = e;
        }
        if (PXmp1_p[jp][0]*PXmp1_p[jp][1]*PXmp1_p[jp][2]==0 && jp!=1) { // par1 is 0 for muons
            cout    << "Warning in readshowers, p" << jp << "mp1 = " << PXmp1_p[jp][0] << " " << PXmp1_p[jp][1] << " " << PXmp1_p[jp][2] << endl;
            outfile << "Warning in readshowers, p" << jp << "mp1 = " << PXmp1_p[jp][0] << " " << PXmp1_p[jp][1] << " " << PXmp1_p[jp][2] << endl;
            warnings1++;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_p >> e;
            PXmp2_p[jp][ip] = e;
        }
        if (PXmp2_p[jp][0]*PXmp2_p[jp][1]*PXmp2_p[jp][2]==0 && jp!=1) { // par1 is 0 for muons
            cout    << "Warning in readshowers, p" << jp << "mp2 = " << PXmp2_p[jp][0] << " " << PXmp2_p[jp][1] << " " << PXmp2_p[jp][2] << endl;
            outfile << "Warning in readshowers, p" << jp << "mp2 = " << PXmp2_p[jp][0] << " " << PXmp2_p[jp][1] << " " << PXmp2_p[jp][2] << endl;
            warnings1++;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_p >> e;
            PXmp3_p[jp][ip] = e;
        }
        if (PXmp3_p[jp][0]*PXmp3_p[jp][1]*PXmp3_p[jp][2]==0 && jp!=1) { // par1 is 0 for muons 
            cout    << "Warning in readshowers, p" << jp << "mp3 = " << PXmp3_p[jp][0] << " " << PXmp3_p[jp][1] << " " << PXmp3_p[jp][2] << endl;
            outfile << "Warning in readshowers, p" << jp << "mp3 = " << PXmp3_p[jp][0] << " " << PXmp3_p[jp][1] << " " << PXmp3_p[jp][2] << endl;
             warnings1++;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_p >> e;
            PXmp4_p[jp][ip] = e;
        }
        if (PXmp4_p[jp][0]*PXmp4_p[jp][1]*PXmp4_p[jp][2]==0 && jp!=1) { // par1 is 0 for muons
        cout    << "Warning in readshowers, p" << jp << "mp4 = " << PXmp4_p[jp][0] << " " << PXmp4_p[jp][1] << " " << PXmp4_p[jp][2] << endl;
        outfile << "Warning in readshowers, p" << jp << "mp4 = " << PXmp4_p[jp][0] << " " << PXmp4_p[jp][1] << " " << PXmp4_p[jp][2] << endl;
            warnings1++;
            return 1;
        }
    }
    asciifile_p.close();

    // Initialize the inverse matrix for the cubic interpolation
    // ---------------------------------------------------------
    InitInverse4by4 ();

    // Flux and derivatives lookup table.
    // Obtain parameters in 100x100 grid of energy and theta values
    // ------------------------------------------------------------
    // debugging checks below, ignore next 12 lines
    /*
    Y[0] = 1;
    Y[1] = 4;
    Y[2] = 9;
    Y[3] = 16;
    double val = solvecubic_mg(0,20.,(1.-0.5)*thetamax/4.,0);
    cout << val << endl;
    Y[0] = 0.3253;
    Y[1] = 0.3205;
    Y[2] = 0.3097;
    Y[3] = 0.2909;
    val = solvecubic_mg(0,20.,0.137511,0);
    cout << val << endl;
    */

    for (int ie=0; ie<100; ie++) {
        double energy = 0.1 + 0.1*ie;

        // Convert energy into the function we use in the interpolation
        // ------------------------------------------------------------
        double xe = 0.5+20.*(log(energy)-log_01)/logdif; // energy is in PeV
        double xe2 = xe*xe;
        for (int it=0; it<100; it++) {
            double theta = it*thetamax/99.; // Want to get 0 and 65 degrees
            Y[0] = exp(PXmg1_p[0][0]) + exp(PXmg1_p[0][1]*pow(xe,PXmg1_p[0][2]));
            Y[1] = exp(PXmg2_p[0][0]) + exp(PXmg2_p[0][1]*pow(xe,PXmg2_p[0][2]));
            Y[2] = exp(PXmg3_p[0][0]) + exp(PXmg3_p[0][1]*pow(xe,PXmg3_p[0][2]));
            Y[3] = exp(PXmg4_p[0][0]) + exp(PXmg4_p[0][1]*pow(xe,PXmg4_p[0][2]));
            thisp0_mg[ie][it]     = solvecubic_mg(0,energy,theta,0);
#ifdef PLOTS
            P0mg->SetBinContent(ie+1,it+1,thisp0_mg[ie][it]);
#endif
            dthisp0de_mg[ie][it]  = solvecubic_mg(0,energy,theta,2);
            d2thisp0de2_mg[ie][it] = solvecubic_mg(0,energy,theta,22);
            d3thisp0de3_mg[ie][it] = solvecubic_mg(0,energy,theta,25);
            dthisp0dth_mg[ie][it] = solvecubic_mg(0,energy,theta,3);
            d2thisp0dth2_mg[ie][it] = solvecubic_mg(0,energy,theta,32);

            Y[0] = PXmg1_p[2][0] + PXmg1_p[2][1]*xe + PXmg1_p[2][2]*xe2;
            Y[1] = PXmg2_p[2][0] + PXmg2_p[2][1]*xe + PXmg2_p[2][2]*xe2;
            Y[2] = PXmg3_p[2][0] + PXmg3_p[2][1]*xe + PXmg3_p[2][2]*xe2;
            Y[3] = PXmg4_p[2][0] + PXmg4_p[2][1]*xe + PXmg4_p[2][2]*xe2;
            thisp2_mg[ie][it]     = solvecubic_mg(2,energy,theta,0);
#ifdef PLOTS
            P2mg->SetBinContent(ie+1,it+1,thisp2_mg[ie][it]);
#endif
            dthisp2de_mg[ie][it]  = solvecubic_mg(2,energy,theta,2);
            d2thisp2de2_mg[ie][it] = solvecubic_mg(2,energy,theta,22);
            d3thisp2de3_mg[ie][it] = solvecubic_mg(2,energy,theta,25);
            dthisp2dth_mg[ie][it] = solvecubic_mg(2,energy,theta,3);
            d2thisp2dth2_mg[ie][it] = solvecubic_mg(2,energy,theta,32);

            Y[0] = PXeg1_p[0][0] * exp(PXeg1_p[0][1] * pow(xe,PXeg1_p[0][2]));
            Y[1] = PXeg2_p[0][0] * exp(PXeg2_p[0][1] * pow(xe,PXeg2_p[0][2]));
            Y[2] = PXeg3_p[0][0] * exp(PXeg3_p[0][1] * pow(xe,PXeg3_p[0][2]));
            Y[3] = PXeg4_p[0][0] * exp(PXeg4_p[0][1] * pow(xe,PXeg4_p[0][2]));
            thisp0_eg[ie][it]     = solvecubic_eg(0,energy,theta,0);
#ifdef PLOTS
            P0eg->SetBinContent(ie+1,it+1,thisp0_eg[ie][it]);
#endif
            dthisp0de_eg[ie][it]  = solvecubic_eg(0,energy,theta,2);
            d2thisp0de2_eg[ie][it] = solvecubic_eg(0,energy,theta,22);
            d3thisp0de3_eg[ie][it] = solvecubic_eg(0,energy,theta,25);
            dthisp0dth_eg[ie][it] = solvecubic_eg(0,energy,theta,3);
            d2thisp2dth2_eg[ie][it] = solvecubic_eg(0,energy,theta,32);
            
            Y[0] = PXeg1_p[1][0] + PXeg1_p[1][1]*xe + PXeg1_p[1][2]*xe2;
            Y[1] = PXeg2_p[1][0] + PXeg2_p[1][1]*xe + PXeg2_p[1][2]*xe2;
            Y[2] = PXeg3_p[1][0] + PXeg3_p[1][1]*xe + PXeg3_p[1][2]*xe2;
            Y[3] = PXeg4_p[1][0] + PXeg4_p[1][1]*xe + PXeg4_p[1][2]*xe2;
            thisp1_eg[ie][it]     = solvecubic_eg(1,energy,theta,0);
#ifdef PLOTS
            P1eg->SetBinContent(ie+1,it+1,thisp1_eg[ie][it]);
#endif
            dthisp1de_eg[ie][it]  = solvecubic_eg(1,energy,theta,2);
            d2thisp1de2_eg[ie][it] = solvecubic_eg(1,energy,theta,22);
            d3thisp1de3_eg[ie][it] = solvecubic_eg(1,energy,theta,25);
            dthisp1dth_eg[ie][it] = solvecubic_eg(1,energy,theta,3);

            Y[0] = PXeg1_p[2][0] + PXeg1_p[2][1]*xe + PXeg1_p[2][2]*xe2;
            Y[1] = PXeg2_p[2][0] + PXeg2_p[2][1]*xe + PXeg2_p[2][2]*xe2;
            Y[2] = PXeg3_p[2][0] + PXeg3_p[2][1]*xe + PXeg3_p[2][2]*xe2;
            Y[3] = PXeg4_p[2][0] + PXeg4_p[2][1]*xe + PXeg4_p[2][2]*xe2;
            thisp2_eg[ie][it]     = solvecubic_eg(2,energy,theta,0);
#ifdef PLOTS
            P2eg->SetBinContent(ie+1,it+1,thisp2_eg[ie][it]);
#endif
            dthisp2de_eg[ie][it]  = solvecubic_eg(2,energy,theta,2);
            d2thisp2de2_eg[ie][it] = solvecubic_eg(2,energy,theta,22);
            d3thisp2de3_eg[ie][it] = solvecubic_eg(2,energy,theta,25);
            dthisp2dth_eg[ie][it] = solvecubic_eg(2,energy,theta,3);
            d2thisp2dth2_eg[ie][it] = solvecubic_eg(2,energy,theta,32);

            Y[0] = exp(PXmp1_p[0][0]) + exp(PXmp1_p[0][1]*pow(xe,PXmp1_p[0][2]));
            Y[1] = exp(PXmp2_p[0][0]) + exp(PXmp2_p[0][1]*pow(xe,PXmp2_p[0][2]));
            Y[2] = exp(PXmp3_p[0][0]) + exp(PXmp3_p[0][1]*pow(xe,PXmp3_p[0][2]));
            Y[3] = exp(PXmp4_p[0][0]) + exp(PXmp4_p[0][1]*pow(xe,PXmp4_p[0][2]));
            thisp0_mp[ie][it]     = solvecubic_mp(0,energy,theta,0);
#ifdef PLOTS
            P0mp->SetBinContent(ie+1,it+1,thisp0_mp[ie][it]);
#endif
            dthisp0de_mp[ie][it]  = solvecubic_mp(0,energy,theta,2);
            dthisp0dth_mp[ie][it] = solvecubic_mp(0,energy,theta,3);

            Y[0] = PXmp1_p[2][0] + PXmp1_p[2][1]*xe + PXmp1_p[2][2]*xe2;
            Y[1] = PXmp2_p[2][0] + PXmp2_p[2][1]*xe + PXmp2_p[2][2]*xe2;
            Y[2] = PXmp3_p[2][0] + PXmp3_p[2][1]*xe + PXmp3_p[2][2]*xe2;
            Y[3] = PXmp4_p[2][0] + PXmp4_p[2][1]*xe + PXmp4_p[2][2]*xe2;
            thisp2_mp[ie][it]     = solvecubic_mp(2,energy,theta,0);
#ifdef PLOTS
            P2mp->SetBinContent(ie+1,it+1,thisp2_mp[ie][it]);
#endif
            dthisp2de_mp[ie][it]  = solvecubic_mp(2,energy,theta,2);
            dthisp2dth_mp[ie][it] = solvecubic_mp(2,energy,theta,3);

            Y[0] = exp(PXep1_p[0][0]) + exp(PXep1_p[0][1]*pow(xe,PXep1_p[0][2]));
            Y[1] = exp(PXep2_p[0][0]) + exp(PXep2_p[0][1]*pow(xe,PXep2_p[0][2]));
            Y[2] = exp(PXep3_p[0][0]) + exp(PXep3_p[0][1]*pow(xe,PXep3_p[0][2]));
            Y[3] = exp(PXep4_p[0][0]) + exp(PXep4_p[0][1]*pow(xe,PXep4_p[0][2]));
            thisp0_ep[ie][it]     = solvecubic_ep(0,energy,theta,0);
#ifdef PLOTS
            P0ep->SetBinContent(ie+1,it+1,thisp0_ep[ie][it]);
#endif
            dthisp0de_ep[ie][it]  = solvecubic_ep(0,energy,theta,2);
            dthisp0dth_ep[ie][it] = solvecubic_ep(0,energy,theta,3);

            Y[0] = PXep1_p[1][0] + PXep1_p[1][1]*xe + PXep1_p[1][2]*xe2;
            Y[1] = PXep2_p[1][0] + PXep2_p[1][1]*xe + PXep2_p[1][2]*xe2;
            Y[2] = PXep3_p[1][0] + PXep3_p[1][1]*xe + PXep3_p[1][2]*xe2;
            Y[3] = PXep4_p[1][0] + PXep4_p[1][1]*xe + PXep4_p[1][2]*xe2;
            thisp1_ep[ie][it]     = solvecubic_ep(1,energy,theta,0);
#ifdef PLOTS
            P1ep->SetBinContent(ie+1,it+1,thisp1_ep[ie][it]);
#endif
            dthisp1de_ep[ie][it]  = solvecubic_ep(1,energy,theta,2);
            dthisp1dth_ep[ie][it] = solvecubic_ep(1,energy,theta,3);

            Y[0] = PXep1_p[2][0] + PXep1_p[2][1]*xe + PXep1_p[2][2]*xe2;
            Y[1] = PXep2_p[2][0] + PXep2_p[2][1]*xe + PXep2_p[2][2]*xe2;
            Y[2] = PXep3_p[2][0] + PXep3_p[2][1]*xe + PXep3_p[2][2]*xe2;
            Y[3] = PXep4_p[2][0] + PXep4_p[2][1]*xe + PXep4_p[2][2]*xe2;
            thisp2_ep[ie][it]     = solvecubic_ep(2,energy,theta,0);
#ifdef PLOTS
            P2ep->SetBinContent(ie+1,it+1,thisp2_ep[ie][it]);
#endif
            dthisp2de_ep[ie][it]  = solvecubic_ep(2,energy,theta,2);
            dthisp2dth_ep[ie][it] = solvecubic_ep(2,energy,theta,3);
        }
    }
    return 0;
}

// Determine the measured gamma fraction and a lower bound on its variance
// -----------------------------------------------------------------------
double MeasuredGammaFraction () {

    // We want to find the MLE of fg in a likelihood which is 
    // L = Prod_i [ fg*pg + (1-fg)*pp ]
    // ln L = Sum_i { log[fg*pg+(1-fg)*pp] }
    // dlnL/dfg = Sum_i { (pg-pp)/[fg*pg+(1-fg)*pp] }
    // d2lnL/dfg2 = Sum_i { -(pg-pp)^2/[fg*pg+(1-fg)*pp)^2]} = -ingsigfg2
    // ------------------------------------------------------------------
    double MeasFg = 0.5;
    double dlnL_dfg_orig, dlnL_dfg; // dlnL_dfg_new;
    double num; 
    double den;
    int Nloops = 0;
    do {
        dlnL_dfg     = 0.;
        inv_sigmafs2 = 0.;
        for (int k=Nevents; k<Nevents+Nbatch; k++) {
            if (Active[k]) {
                num = pg[k]-pp[k];
                den = (MeasFg*pg[k]+(1.-MeasFg)*pp[k]);
                if (den>0.) {
                    dlnL_dfg     += PActive[k]*num/den;
                    inv_sigmafs2 += PActive[k]*num*num/(den*den);
                }
            }
        }
        if (Nloops==0) dlnL_dfg_orig = dlnL_dfg;
        if (inv_sigmafs2==0.) {
            inv_sigmafs2 = epsilon;
            cout    << "Warning, inv_sigmafs2 is zero. " << endl;
            outfile << "Warning, inv_sigmafs2 is zero. " << endl;
            warnings6++;
        }

        // Since we get dlnL/dfg = x !=0, and we want it to be =0, we need to modify MeasFg.
        // The variation dMeasFg is found by noticing that we want dlnL/dMeasFg to change by -x,
        // and such a change for a change of dMeasFg equals the second derivative dlnL/dMeasFg,
        // i.e. minus the inverse of sigmafg^2. So -x/dMeasFg = -1/sigmaMeasFg^2 from which we
        // get the expression below.
        // -----------------------------------------------------------------------------
        MeasFg += dlnL_dfg/inv_sigmafs2;

        Nloops++;
        // cout << Nloops << " dlnl_dfg = " << dlnL_dfg << " " << MeasFg << endl;
    } while (fabs(dlnL_dfg/inv_sigmafs2)>0.000001);
    // cout << "     After " << Nloops << " loops, calc of measured Fg gives " << MeasFg 
    //      << " with dlnL/dFg = " << dlnL_dfg_new << " (for 0.5 was " << dlnL_dfg_orig << ")" << endl;
    return MeasFg;
}

// This routine is called after a set of detector moves. It ensures units are separated by a minimum amount
// --------------------------------------------------------------------------------------------------------
void ResolveOverlaps() {
    double d2min = largenumber;
    double D2ij;
    int indmini  = 0;
    int indminj  = 1;
    int unitspersector = Nunits/multiplicity; // Account for CommonMode>=2 cases
    for (int i=0; i<unitspersector-1; i++) {
        for (int j=i+1; j<unitspersector; j++) {
            D2ij = pow(x[i]-x[j],2.) + pow(y[i]-y[j],2.);
            if (D2ij<d2min) {
                d2min = D2ij;
                indmini = i;
                indminj = j;
            }
        }
    }
    if (d2min>=DefaultR2min) return; // Nothing to be done
    do {
        // cout << "     Relocating units i,j = " << indmini << ", " << indminj << " as Dij = " << sqrt(d2min) << endl;

        // First of all ensure that the closest pair is not exactly on the same spot
        // (and if so randomly choose a direction for minimum displacement)
        // -------------------------------------------------------------------------
        double ijangle;
        if (d2min<=0.) {
            ijangle = myRNG->Uniform(-0.5*pi,0.5*pi);
        } else if (x[indmini]==x[indminj]) {
            ijangle = 0.5*pi;
        } else {
            ijangle = atan((y[indminj]-y[indmini])/(x[indminj]-x[indmini]));
        }

        // Displace the two units by minimum amount 
        // ----------------------------------------
        double delta = 0.5*(sqrt(DefaultR2min)-sqrt(d2min))+0.01; // Make sure the new positions satisfy requirement
        if (x[indmini]==x[indminj]) {
            if (y[indminj]>y[indmini]) {
                y[indminj] = y[indminj] + delta;
                y[indmini] = y[indmini] - delta;
            } else {
                y[indminj] = y[indminj] - delta;
                y[indmini] = y[indmini] + delta;
            }
        } else if (x[indminj]>x[indmini]) {
            x[indminj] = x[indminj] + delta * cos(ijangle);
            y[indminj] = y[indminj] + delta * sin(ijangle);
            x[indmini] = x[indmini] - delta * cos(ijangle);
            y[indmini] = y[indmini] - delta * sin(ijangle);
        } else {
            x[indminj] = x[indminj] - delta * cos(ijangle);
            y[indminj] = y[indminj] - delta * sin(ijangle);
            x[indmini] = x[indmini] + delta * cos(ijangle);
            y[indmini] = y[indmini] + delta * sin(ijangle);
        }
        // Recompute all relevant D2ij elements
        // ------------------------------------
        d2min = largenumber;
        int indmini_new = 0;
        int indminj_new = 1;
        for (int j=1; j<unitspersector; j++) {
            if (j!=indmini) {
                D2ij = pow(x[indmini]-x[j],2.) + pow(y[indmini]-y[j],2.); // New distance
                if (D2ij<d2min) {
                    d2min = D2ij;
                    indmini_new = indmini;
                    indminj_new = j;
                }
            }
        }
        for (int i=0; i<unitspersector-1; i++) {
            if (i!=indminj) {
                D2ij = pow(x[i]-x[indminj],2.) + pow(y[i]-y[indminj],2.); // New distance 
                if (D2ij<d2min) {
                    d2min = D2ij;
                    indmini_new = i;
                    indminj_new = indminj;
                }
            }
        }
        if (d2min<DefaultR2min) {
            indmini = indmini_new;
            indminj = indminj_new;
        }
    } while (d2min<DefaultR2min); // Continue relocating units until all are above min distance

    // If multiplicity>1, we relocate all other units accordingly
    // ----------------------------------------------------------
    if (multiplicity>1) {
        for (int id=0; id<unitspersector; id++) {
            double r = 0.;
            r = pow(x[id],2)+pow(y[id],2);
            if (r>0.) r = sqrt(r);
            double phi = PhiFromXY (x[id],y[id]);
            for (int itr=1; itr<multiplicity; itr++) {
                x[id+itr*unitspersector] = r*cos(phi+itr*twopi/multiplicity); 
                y[id+itr*unitspersector] = r*sin(phi+itr*twopi/multiplicity); 
            }
        }
    }
    // Job done, return to caller
    // --------------------------
    return;
}

// This function prevents detectors from occupying specific regions on the ground
// ------------------------------------------------------------------------------
void ForbiddenRegion (int type, int idstar, int mode) {
    if (mode==0) { // Relocate detectors that have straggled into forbidden region
        bool MovedOne = false;
        for (int id = 0; id<Nunits; id++) {
            if (type==0) { // The forbidden region is a circle of center xc,yc and radius FRpar[2]
                float xc      = FRpar[0];
                float yc      = FRpar[1];
                float radius  = FRpar[2];
                if ((!IncludeOnlyFR && pow(x[id]-xc,2.)+pow(y[id]-yc,2.)<radius*radius) || 
                    (IncludeOnlyFR  && pow(x[id]-xc,2.)+pow(y[id]-yc,2.)>radius*radius)) {
                    // Move detector to boundary
                    // -------------------------
                    double phi = PhiFromXY(x[id]-xc,y[id]-yc);
                    x[id] = xc+radius*cos(phi);
                    y[id] = yc+radius*sin(phi);
                    // Disengage this multiplet if CommonMode>=2 - WORK IN PROGRESS
                    // MultipletIndex is a sequential index for the multiplet containing detector id
                    // ------------------------------------------------------------
                    // int MultipletIndex = id/(Nunits/CommonMode);
                    // Disengaged[MultipletIndex] = true;
                    InMultiplet[id] = false;
                    MovedOne = true;
                }
            } else if (type==1) { // Semiplane not containing origin, not vertically aligned
                float m = FRpar[0];
                float q = FRpar[1];
                if (!IncludeOnlyFR) {
                    if (y[id]>m*x[id]+q) {
                        // Move detector to boundary
                        // -------------------------
                        float d =  fabs(m*x[id]-y[id]+q)/sqrt(m*m+1.); // Distance from line
                        float theta = atan(m);
                        x[id] = x[id] + d*sin(theta);
                        y[id] = y[id] - d*cos(theta);
                        InMultiplet[id] = false;
                        MovedOne = true;
                    } 
                } else {
                    if (y[id]<m*x[id]+q) {
                        // Move detector to boundary
                        // -------------------------
                        float d =  fabs(m*x[id]-y[id]+q)/sqrt(m*m+1.); // Distance from line
                        float theta = atan(m);
                        x[id] = x[id] - d*sin(theta);
                        y[id] = y[id] + d*cos(theta);
                        InMultiplet[id] = false;
                        MovedOne = true;
                    } 
                }
            } else if (type==2) {
                float xmin = FRpar[0];
                float xmax = FRpar[1];
                if (IncludeOnlyFR && x[id]>xmin && x[id]<xmax) {
                    // Move to boundary
                    // ----------------
                    if (fabs(x[id]-xmin)<fabs(x[id]-xmax)) {
                        x[id] = xmin;
                        InMultiplet[id] = false;
                        MovedOne = true;
                    } else {
                        x[id] = xmax;
                        InMultiplet[id] = false;
                        MovedOne = true;
                    }
                } else if (!IncludeOnlyFR && (x[id]<xmin || x[id]>xmax)) {
                    if (x[id]<xmin) {
                        x[id] = xmin;
                        InMultiplet[id] = false;
                        MovedOne = true;
                    } else {
                        x[id] = xmax;
                        InMultiplet[id] = false;
                        MovedOne = true;
                    }
                }
            } 
        }
        return;
    } else if (mode==1) { // Generate repulsive derivative
        dU_dxiFR = 0.;
        dU_dyiFR = 0.;
        if (type==0) { // Circle
            float xc      = FRpar[0];
            float yc      = FRpar[1];
            float radius  = FRpar[2];
            float deltar  = maxDispl; // To be tuned
            float currentr= sqrt(pow(x[idstar]-xc,2.)+pow(y[idstar]-yc,2.));
            if ((!IncludeOnlyFR && currentr<radius+deltar) || 
                (IncludeOnlyFR  && currentr>radius-deltar)) {
                double dudx = dU_dxi[idstar];
                double dudy = dU_dyi[idstar];
                double du   = sqrt(pow(dudx,2.)+pow(dudy,2.));
                double phi  = PhiFromXY(x[idstar]-xc,y[idstar]-yc);
                double phiU = PhiFromXY(dudx,dudy);
                double frac_repulsion = 1.-fabs(currentr-radius)/deltar;
                double du_repulsive = du*cos(phi-phiU)*frac_repulsion; // Component of gradient toward forbidden region
                if (du_repulsive>0.) { // Zero off the component of the gradient
                    if (!IncludeOnlyFR) {
                        dU_dxiFR = du_repulsive*cos(phi);
                        dU_dyiFR = du_repulsive*sin(phi);
                    } else {
                        dU_dxiFR = -du_repulsive*cos(phi);
                        dU_dyiFR = -du_repulsive*sin(phi);
                    } 
                }
                //cout << "  det # " << idstar << " x,y = " << x[idstar] << " " << y[idstar] << " dudx,dy = " 
                //     << dudx << " " << dudy << " phiU = " << phiU << " phi = " << phi << " du_rep = " << du_repulsive
                //     << " dUdxfr = " << dU_dxiFR << " dUdyfr = " << dU_dyiFR << endl;
            }
        } else if (type==1) { // Band not orthogonal to x axis
            float m = FRpar[0];
            float q = FRpar[1];
            float deltay = maxDispl; // Note, width of repulsion region is defined by its projection on y axis for now
            if ((!IncludeOnlyFR && y[idstar]>m*x[idstar]+q-deltay) || 
                (IncludeOnlyFR && y[idstar]<m*x[idstar]+q+deltay)) { // We only generate repulsion close to boundary
                // Generate repulsive derivative
                // -----------------------------
                double dudx = dU_dxi[idstar];
                double dudy = dU_dyi[idstar];
                double du   = sqrt(pow(dudx,2.)+pow(dudy,2.));
                double phi;
                if (!IncludeOnlyFR) {
                    phi = atan(m)+0.5*pi; // Angle of normal to line delimiting forbidden semiplane, pointing toward it
                } else {
                    phi = atan(m)-0.5*pi; // Normal toward forbidden region is downward if we are staying above the line
                }
                double phiU = PhiFromXY(dudx,dudy);
                double frac_repulsion = 1.-fabs(y[idstar]-m*x[idstar]-q)/deltay;
                double du_repulsive = du*cos(phi-phiU)*frac_repulsion; // Component of gradient toward forbidden region
                if (du_repulsive>0.) { // We want to zero this off
                    // Now decompose repulsive force into x and y gradients
                    dU_dxiFR = -du_repulsive*cos(phi);
                    dU_dyiFR = -du_repulsive*sin(phi);
                }
            }  
        } else if (type=2) { // Band in x
            float xmin = FRpar[0];
            float xmax = FRpar[1];
            float deltax = maxDispl;
            double frac_repulsion;
            // Case 1: the detector is close to the boundary, at higher x
            if ((IncludeOnlyFR  && (x[idstar]>=xmax && x[idstar]<xmax+deltax)) || 
                (!IncludeOnlyFR && (x[idstar]>=xmin && x[idstar]<xmin+deltax)) ) {                
                if (x[idstar]>=xmax) frac_repulsion = 1.-fabs(x[idstar]-xmax)/deltax;
                if (x[idstar]<xmin+deltax) frac_repulsion = 1.-fabs(x[idstar]-xmin)/deltax;
                if (dU_dxi[idstar]<0.) dU_dxiFR = -dU_dxi[idstar]*frac_repulsion; // Nullify x gradient
            } else if ((!IncludeOnlyFR && (x[idstar]<xmax && x[idstar]>xmax-deltax)) || 
                       (IncludeOnlyFR  && (x[idstar]<xmin && x[idstar]>xmin-deltax))) {
                if (x[idstar]<xmin) frac_repulsion = 1.-fabs(x[idstar]-xmin)/deltax;
                if (x[idstar]>xmax-deltax) frac_repulsion = 1.-fabs(x[idstar]-xmax)/deltax;
                if (dU_dxi[idstar]>0.) dU_dxiFR = -dU_dxi[idstar]*frac_repulsion; // Nullify x gradient
            }
        }
        return;
    } 
    return;
}

// Recenter array before enforcing forbidden region
// ------------------------------------------------
void RecenterArray () {
    float xave = 0.;
    float yave = 0.;
    for (int id = 0; id<Nunits; id++ ) {
        xave += x[id];
        yave += y[id];
    }
    xave = xave/Nunits;
    yave = yave/Nunits;
    for (int id=0; id<Nunits; id++) {
        x[id] = x[id] - xave;
        y[id] = y[id] - yave;
    }
    cout    << "     Recentered array by moving it " << xave << "m down and " << yave << "m left." << endl;
    outfile << "     Recentered array by moving it " << xave << "m down and " << yave << "m left." << endl;
    return;
}

// The function below computes the minimum distance allowed between tank macroaggregates consisting of N tanks
// arranged in hexagonal structure. This is computed by comparing N to the max number of tanks that can be
// arranged in successive circles (1,7,19,37,60,91,...), and using the criterion that the two macroaggregates
// must be spaced as individual tanks (minTankSpacing).
// -----------------------------------------------------------------------------------------------------------
double D2min (int N) {
    int k = 0;
    if (N<=0) return 0;
    int Ncurr;
    do {
        Ncurr = 1 + 3*(k*k+k);
        k++;
        //cout << " k " << k << " Nmax = " << Ncurr << " Dmin1 = " << (2*k-1)*(2.*TankRadius+minTankSpacing) << " Dmin2 = " << 2.*((2*k-1)*TankRadius+(2*k-1)*minTankSpacing/2.) << endl;
    } while (Ncurr<N);
    return pow((2*k-1)*(2.*TankRadius+minTankSpacing),2.); 
    // Alternative definition also working:
    // return pow(2.*((2*k-1)*TankRadius+(2*k-1)*minTankSpacing/2.),2.);
}

// Importance sampling routine (example)
// -------------------------------------
bool ImportanceSample (int Nmuons) {
    double prob = exp(-0.1*Nmuons); // This can be configured to be different, depending on how strongly you want to select muon starved events
    double r = myRNG->Uniform();
    if (r<prob) return true;
    return false;
}
double Weight_IS (int Nmuons) {
    return 1./exp(-0.1*Nmuons);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//                                                              Main routine
//
// -------------------------------------------------------------------------------------------------------------------------------------
#if defined(STANDALONE) || defined(UBUNTU)
int main (int argc, char * argv[]) {

    // Default values of pass parameters (same in root routine pass par list, see below)
    // ---------------------------------------------------------------------------------
    Nevents            = 2000;
    Nbatch             = Nevents;
    Nunits             = 169;
    Nepochs            = 2000;
    shape              = 3;
    CommonMode         = 3;
    DetectorSpacing    = 30;
    SpacingStep        = 30;
    Ntrigger           = 50;
    plotBitmap         = 14335; // 14271; -> added 64
    DisplFactor        = 2.;
    addSysts           = false; // This is the default so that if -sys is used the systematics are turned on
    RelResCounts       = 0.05;
    GenGammaFrac       = 0.5;
    // Other parameters
    // ----------------
    Rslack             = 2000; // Updated from 2500 since v123
    StartLR            = 1.0;
    Ngrid              = 100;
    NEgrid             = 10;
    Nsteps             = 500;
    LRX                = 1.0;
    LRE                = 0.05;
    LRA                = 0.05;
    Eslope             = 0.;     // 0. is default for PeVSource
    double eta_GF_mult = 1.;
    double eta_IR_mult = 1.;    
    double eta_PR_mult = 1.;
    double eta_PS_mult = 1.;
    double eta_TA_mult = 1.;
    double eta_TL_mult = 1.;
    startEpoch         = 0.;
    TankNumber         = 19;
    OrthoShowers       = false;
    SlantedShowers     = false;
    usetrueXY          = false;
    usetrueAngs        = false;
    usetrueE           = false;
    scanU              = false;
    readGeom           = false;
    PredefinedLayout   = true;  // only used if readGeom is true
    PredefinedLayoutID = 0;     // ditto
    noSGDupdate        = false;
    initTrueVals       = true;
    Nthreads           = 32;
    UseAreaCost        = false;
    UseLengthCost      = false;
    PeVSource          = false;
    E_PS               = 2.;
    Bgr_mu_per_m2      = 0.000001826*IntegrationWindow;
    Bgr_e_per_m2       = 0.000000200*IntegrationWindow;
    VoidRegion         = false;
    fixE               = false;
#ifdef RUNBENCHMARK
    noSGDupdate        = true;
    Nevents            = 2000;
    Nbatch             = 2000;
    Nepochs            = 10;
    CommonMode         = 0;
    eta_GF_mult        = 1.;
    eta_IR_mult        = 1.;
    eta_PR_mult        = 1.;
    eta_PS_mult        = 1.;
    eta_TA_mult        = 1.;
    eta_TL_mult        = 1.;
    UseAreaCost        = false;
    UseLengthCost      = false;
    Nthreads           = 6; // With this, can run 5 jobs in parallel on one 32-core machine
    TankNumber         = 1;
    plotBitmap         = 14271;   
    PeVSource          = true;
    E_PS               = 2.;
    useN5s             = true;
    RndmSeed           = true; // To generate different sequences at every run
#endif
    // Command line arguments
    // ----------------------
    for (int i=0; i<argc; i++) {
        if (!strcmp(argv[i],"-h")) {
            cout << "List of arguments:" << endl;
            cout << "-nev number of events for PDF generation (expects int)" << endl;
            cout << "-nba number of batch events (expects int)" << endl;
            cout << "-ggf fraction of gammas in batches" << endl;
            cout << "-nde number of detector units (expects int)" << endl;
            cout << "-nep number of training epochs (expects int)" << endl;
            cout << "-spa spacing between detectors in meters (expects float)" << endl;
            cout << "-sst step between elements in meters (expects float)" << endl;
            cout << "-dis displacement factor" << endl;
            cout << "-rsl extra radial space in generated shower area (expects float)" << endl;
            cout << "-sha shape of initial layout (0-9, 101-114) (expects int)" << endl;
            cout << "-com common mode (0 independent movement, 1,2 common modes)" << endl;
            cout << "-slr start learning rate (expects float in 0.1:10.)" << endl;
            cout << "-ngr number of grid search points in xy in shower likelihood (expects int)" << endl;
            cout << "-nei number of energy initializations (expects int)" << endl;
            cout << "-nst max number of steps in shower likelihood (expects int)" << endl;
            cout << "-lrx learning rate for position (expects float)" << endl;
            cout << "-lre learning rate for energy (expects float)" << endl;
            cout << "-lra learning rate for angles (expects float)" << endl;
            cout << "-esl slope of energy distribution of showers (def. 0)" << endl;
            cout << "-etf utility gradient multiplier of flux uncertainty (def. 1.)" << endl;
            cout << "-ete utility gradient multiplier of integrated energy resolution (def. 1.)" << endl;
            cout << "-etp utility gradient multiplier of pointing resolution (def. 1.)" << endl;
            cout << "-ets utility gradient multiplier of point source (def. 1.)" << endl;
            cout << "-eta utility gradient multiplier of area cost (def. 1.)" << endl;
            cout << "-etl utility gradient multiplier of length cost (def. 1.)" << endl;
            cout << "-ste starting epoch (use to continue previous runs, expects int)" << endl;
            cout << "-ntr minimum number of detectors hit by triggering showers" << endl;
            cout << "-tnu tank number in macroaggregates (for tank area calculation)" << endl;
            cout << "-sys add systematics smearing to counting in tanks" << endl;
            cout << "-rrc relative resolution on particle counts (def. 0.05 if sys is on)" << endl;
            cout << "-utx use true (pbmand do not fit) XY of showers" << endl;
            cout << "-ute use true (and do not fit) energy of showers" << endl;
            cout << "-uta use true (and do not fit) angle of showers" << endl;
            cout << "-ort use orthogonal showers (theta=0, phi undef)" << endl;
            cout << "-sla use showers at theta=pi/4" << endl;
            cout << "-sca scan utility function around one point" << endl;
            cout << "-rea read geometry from previous run" << endl;
            cout << "-pli predefined layout ID" << endl;
            cout << "-nsu do NOT do SGD updates" << endl;
            cout << "-ntv do NOT use true shower parameters for initialization of likelihood reco" << endl;
            cout << "-nth number of threads (expects int)" << endl;
            cout << "-pbm bitmap for graph plotting choice" << endl;
            cout << "-uac use area cost" << endl;
            cout << "-ulc use length cost" << endl;
            cout << "-pso use pev source to optimize layout" << endl;
            cout << "-pse energy of source" << endl;
            cout << "-bmu muon background per m2" << endl;
            cout << "-bel electron background per m2" << endl;
            cout << "-voi void a region outside Pampa la Bola triangle (or others, dep. on code mods.)" << endl;
            cout << "-u3s use 3-sigma criterion for PeV source utility" << endl;
            cout << "-fxe fix energy to predetermined value" << endl;
            return 0;
        }
        else if (!strcmp(argv[i],"-nev")) {Nevents         = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-nba")) {Nbatch          = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-ggf")) {GenGammaFrac    = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-nde")) {Nunits          = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-nep")) {Nepochs         = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-spa")) {DetectorSpacing = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-sst")) {SpacingStep     = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-dis")) {DisplFactor     = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-rsl")) {Rslack          = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-sha")) {shape           = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-com")) {CommonMode      = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-slr")) {StartLR         = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-ngr")) {Ngrid           = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-nei")) {NEgrid          = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-nst")) {Nsteps          = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-lrx")) {LRX             = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-lre")) {LRE             = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-lra")) {LRA             = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-esl")) {Eslope          = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-etf")) {eta_GF_mult     = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-ete")) {eta_IR_mult     = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-etp")) {eta_PR_mult     = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-ets")) {eta_PS_mult     = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-eta")) {eta_TA_mult     = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-etl")) {eta_TL_mult     = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-ste")) {startEpoch      = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-ntr")) {Ntrigger        = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-tnu")) {TankNumber      = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-pbm")) {plotBitmap      = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-sys")) {addSysts        = true;}
        else if (!strcmp(argv[i],"-rrc")) {RelResCounts    = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-utx")) {usetrueXY       = true;}
        else if (!strcmp(argv[i],"-ute")) {usetrueE        = true;}    
        else if (!strcmp(argv[i],"-uta")) {usetrueAngs     = true;}
        else if (!strcmp(argv[i],"-ort")) {OrthoShowers    = true;}    
        else if (!strcmp(argv[i],"-sla")) {SlantedShowers  = true;}    
        else if (!strcmp(argv[i],"-sca")) {scanU           = true;}    
        else if (!strcmp(argv[i],"-rea")) {readGeom        = true;}
        else if (!strcmp(argv[i],"-pli")) {
                                           PredefinedLayoutID = atoi(argv[++i]);
                                           PredefinedLayout   = true;
        }    
        else if (!strcmp(argv[i],"-nsu")) {noSGDupdate     = true;}    
        else if (!strcmp(argv[i],"-ntv")) {initTrueVals    = false;}            
        else if (!strcmp(argv[i],"-nth")) {Nthreads        = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-pbm")) {plotBitmap      = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-uac")) {UseAreaCost     = true;}    
        else if (!strcmp(argv[i],"-ulc")) {UseLengthCost   = true;}    
        else if (!strcmp(argv[i],"-pso")) {PeVSource       = true;}
        else if (!strcmp(argv[i],"-pse")) {E_PS            = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-bmu")) {Bgr_mu_per_m2   = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-bel")) {Bgr_e_per_m2    = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-voi")) {VoidRegion      = true;} 
        else if (!strcmp(argv[i],"-u3s")) {useN5s          = false;} 
        else if (!strcmp(argv[i],"-fxe")) {fixE            = true;}
    }  
#endif // STANDALONE || UBUNTU

    // Pass parameters:
    // ----------------
    // Nevents          = number of generated showers for templates generation (choose an even number to have same # of p and g showers)
    // Nbatch           = number of showers per batch in gradient descent. Note, this number should usually be equal to Nevents and EVEN 
    // GenGammaFrac     = fraction of gammas in Nbatch events (in first Nevents it is always 0.5)
    // Nunits           = number of detector elements. For radial distr, use 1/7/19/37/61/91/127/169/217/271/331/397/469/547/631/721...
    // Nepochs          = number of SGD loops
    // DetectorSpacing  = initial spacing of tanks
    // SpacingStep      = increase in spacing 
    // DisplFactor      = factor controlling max displacement upon varying detectorspacing and spacingstep
    // Rslack           = space of showers away from detector units
    // shape            = geometry of the initial layout (0=hexagonal, 1=taxi, 2=spiral)
    // CommonMode       = whether xy of units is varied independently (0), or radius (1), or in multiplets at regular phi intervals
    // StartLR          = starting learning rate for grad descent
    // Ngrid            = number of grid points on the plane for initial assay of likelihood value
    // NEgrid           = number of energy points in [Emin,Emax] for initial assay of likelihood value
    // Nsteps           = max number of steps in likelihood reconstruction
    // LRX              = learning rate multiplier of dlogL/dX in likelihood maximization
    // LRE              = learning rate multiplier of dlogL/dE in likelihood maximization
    // LRA              = learning rate multiplier of dlogL/dth, ph in likelihood maximization
    // Eslope           = slope of energy distribution of showers
    // eta_GF_mult      = utility gradient multiplier for gamma flux uncertainty (defaults to 1)
    // eta_IR_mult      = utility gradient multiplier for integrated energy resolution (def. 1)
    // eta_PR_mult      = utility gradient multiplier for position resolution (def. 1)
    // eta_PS_mult      = utility gradient multiplier for point source utility (def. 1)
    // eta_TA_mult      = utility gradient multiplier for area resolution (def. 1)
    // eta_TL_mult      = utility gradient multiplier for length resolution (def. 1)
    // startEpoch       = starting epoch (for continuing runs with readGeom true, to not overwrite plots)
    // Ntrigger         = minimum number of detectors hit by accepted shower
    // TankNumber       = number of tanks in aggregates 
    // addSysts         = add systematic uncertainty to particle counts
    // RelResCounts     = relative resolution on particle counts
    // usetrueXY        = if true xy of showers is not fit, true values used 100
    // usetrueE         = if true energy of showers is not fit, true value used
    // usetrueAngs      = if true angles are not fit, true value used
    // OrthoShowers     = if true, showers are generated with zero polar angle 
    // SlantedShowers   = if true, showers are generated at pi/4
    // scanU            = if true, a scan of the utility is performed around one point
    // readGeom         = if true, the geometry is read in from file with corresponding parameters
    // noSGDupdate      = if true, no SGD updates of detector positions is operated (used for U estimates)
    // initTrueVals     = if true, parameters are initialized to true ones for correct hypothesis
    // Nthreads         = number of threads for multi-CPU use
    // plotBitmap       = bitmap for choosing which graphs to plot
    // UnitAreaCost     = whether to use area as cost in utility
    // UnitLengthCost   = whether to use length as cost in utility
    // PeVSource        = whether to use a PeV source significance as optimization metric
    // E_PS             = energy of source in PeV
    // Bgr_mu_per_m2    = background muons per m2
    // Bgr_e_per_m2     = background electrons per m2
    // VoidRegion       = whether to void ground outside a triangle (for Pampa la Bola), or other region defined in code
    // useN5s           = if true, use 5-sigma counting excess for PeV source utility, instead of 3-sigma criterion
    // fixE             = if true, fix energy of showers to Efix
#ifdef INROOT
int swgolo (int nev = 2000, int nde = 60, int nep = 200, int sha = 3, int com = 3,  
            double spa = 30., double sst = 30., int ntr = 50, int pbm = 14335, // 5135, // 14271, 
            double dis = 2., bool sys = false, double rrc = 0.05, int mode = 0, double ggf = 0.5) {
    // UNITS
    // -----
    // position: meters
    // angle:    radians
    // time:     nanoseconds
    // energy:   PeV

    bool dedrtr= false; // Not used anymore
    int nba    = nev;   // Now fixed
    // Other parameters (also fixed in call to main, see above)
    // --------------------------------------------------------
    double rsl = 2000;  // Nb MUST BE 2000. // (updated since v123, used to be 2500)
    double slr = 1.0;    
    int ngr    = 100;
    int nei    = 10;
    int nst    = 500;
    double lrx = 1.;
    double lre = 0.05;
    double lra = 0.05; 
    double esl = 0.0;   // NNBB <---------------------- 0.; see hack below

    int ste    = 0;     
    int tnu    = 19;    // NB default should be 19
    bool ort   = false; // Nnbb false is def; use with sca = true;
    bool sla   = false;
    bool   utx, ute, uta, pso; // Defined below
    bool sca   = false; // NNBB false is def
    bool rea   = false; 
    bool nsu   = false;
    bool ntv   = false;
    int nth    = 1;     // No multithreading if running in root
    double eta = 1.;
    double etl = 1.;
    bool uac   = false;
    bool ulc   = false;
    double pse = 1.;
    double bmu = Bgr_mu_per_m2;
    double bel = Bgr_e_per_m2;
    double etf, ete, etp, ets;

    // Pre-defined standard modes of operation
    // ---------------------------------------
    if (mode!=0) { // We use it to study triggering conditions and other 
                   // ... detail that require no background to be seen better
        bmu = 0.;
        bel = 0.;
    }
    if (mode==0) { // NNBB utx should always be false, or unrealistic likelihood max conditions arise
        etf = 1.;
        ete = 1.;
        etp = 1.;
        ets = 0.;
        utx = false;
        ute = false;
        uta = false;
        pso = false;
    } else if (mode==1) {
        etf = 1.;
        ete = 0.;
        etp = 0.;
        ets = 0.;
        utx = false;
        ute = false;
        uta = false;
        pso = false;
    } else if (mode==2) {
        etf = 0.;
        ete = 1.;
        etp = 0.;
        ets = 0.;
        utx = false;
        ute = false;
        uta = false;
        pso = false;
    } else if (mode==3) {
        etf = 1.;
        ete = 1.;
        etp = 0.;
        ets = 0.;
        utx = false;
        ute = false;
        uta = false;
        pso = false;
    } else if (mode==4) {
        etf = 0.;
        ete = 0.;
        etp = 1.;
        ets = 0.;
        utx = false;
        ute = false;
        uta = false;
        pso = false;
    } else if (mode==5) {
        etf = 1.;
        ete = 0.;
        etp = 1.;
        ets = 0.;
        utx = false;
        ute = false;
        uta = false;
        pso = false;
    } else if (mode==6) {
        etf = 0.;
        ete = 1.;
        etp = 1.;
        ets = 0.;
        utx = false;
        ute = false;
        uta = false;
        pso = false;
    } else if (mode==7) {
        etf = 1.;
        ete = 1.;
        etp = 1.;
        ets = 0.;
        utx = false;
        ute = false;
        uta = false;
        pso = false;
    } else if (mode==8) {
        etf = 1.;
        ete = 1.;
        etp = 1.;
        ets = 1.;
        utx = false;
        ute = false;
        uta = false;
        pso = true;
    }
    double eta_GF_mult     = etf;
    double eta_IR_mult     = ete;
    double eta_PR_mult     = etp;
    double eta_PS_mult     = ets;
    double eta_TA_mult     = eta;
    double eta_TL_mult     = etl;

    // Get static values from pass parameters
    // --------------------------------------
    Nevents                = nev;
    Nbatch                 = nba;
    GenGammaFrac           = ggf;
    Nunits                 = nde;
    Nepochs                = nep;
    DetectorSpacing        = spa;
    SpacingStep            = sst;
    DisplFactor            = dis;
    Rslack                 = rsl;
    shape                  = sha;
    CommonMode             = com;
    StartLR                = slr;
    Ngrid                  = ngr;
    NEgrid                 = nei;
    Nsteps                 = nst;
    LRX                    = lrx;
    LRE                    = lre;
    LRA                    = lra;
    Eslope                 = esl;
    Nthreads               = nth;
    startEpoch             = ste;
    Ntrigger               = ntr;
    TankNumber             = tnu;
    addSysts               = sys;
    RelResCounts           = rrc;
    usetrueXY              = utx;
    usetrueE               = ute;
    usetrueAngs            = uta;
    OrthoShowers           = ort;
    SlantedShowers         = sla;
    scanU                  = sca;
    readGeom               = rea;
    noSGDupdate            = nsu;
    initTrueVals           = !ntv;
    plotBitmap             = pbm;
    UseAreaCost            = uac;
    UseLengthCost          = ulc;
    PeVSource              = pso;
    E_PS                   = pse;
    Bgr_mu_per_m2          = bmu;
    Bgr_e_per_m2           = bel;
    dedrtrue               = dedrtr;
    VoidRegion             = false;
    useN5s                 = true;
    fixE                   = false;
#endif

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Put here any override of parameter definition to specify special runs
    // (rather than hacking the code above!)
    // ---------------------------------------------------------------------
    // Eslope = 1.; // HACK!
    // fixE = true;
    // OrthoShowers = true;
    // SlantedShowers = true;
    // VoidRegion = true;
    // UseLengthCost = true;
    // eta_TL_mult = 1.;
    // UseAreaCost = true;

    // Parameters to study resolutions for A5 inner array
    // --------------------------------------------------
    // Nevents        = 5000;
    // Nbatch         = 5000;
    // noSGDupdate    = true;
    // SlantedShowers  = true;
    // Eslope          = 0.;
    // TankNumber      = 19;
    // Nunits          = 19;
    // Nepochs         = 100;
    // DetectorSpacing = 22.1; // 0.6m spacing between units + 19 units = (1.91+0.6+3.82+0.6+3.82)*2 + 0.6 = 
    // SpacingStep     = 22.1;

    // Parameters to study behaviour at boundaries
    // -------------------------------------------
    // VoidRegion      = true;
    // TankNumber      = 19;
    // PeVSource       = true;
    // E_PS            = 6.;
    // UseAreaCost     = true;
    // UseLengthCost   = false;
    // useN5s          = true; // note, only meaningful if PeVSource is on.
    // E_PS = 0.2;
    
    // Used to study the effect of tank size on flux estimation
    // (which is computed at the center of the tank by default)
    // --------------------------------------------------------
    //StudyFluxRatio = true;

    // Settings to produce scans of utility. Can run with mode=1 or 2
    // to produce maps of utility for GF or IR parts alone 
    // --------------------------------------------------------------
    // scanU       = true;
    // SameShowers = true;
    // fixE     = true;
    // Nunits = 38;
    // idstar = 37; // 37
    // Rslack = 1500;
    // // SlantedShowers = true; // to be used with mode=8 if scanning U
    //fixE         = true;
    //Efix         = 0.2;
    // KeepCentered = true;
    // VoidRegion   = true;
    // UseAreaCost  = false;
    // E_PS         = 6.; // in case we run mode=8 we take this range

    // Settings to run from initial optimized layout
    // ---------------------------------------------
    // readGeom = true;
    // PredefinedLayout   = true;
    // PredefinedLayoutID = 3; // or 4
    // shape              = 3;
    // CommonMode         = 3;
    // eta_GF_mult        = 0;
    // PeVSource          = false;
    // Nunits             = 36;
    // TankNumber         = 19;
    // VoidRegion         = true;
    // addSysts           = false;
    // UseAreaCost        = false;
    // UseLengthCost      = false;
    // Eslope             = 0;
    // eta_IR_mult        = 1.;
    // eta_PR_mult        = 1.;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Reset the coefficient of gradient descent terms
    // -----------------------------------------------
    eta_GF *= eta_GF_mult;
    eta_IR *= eta_IR_mult;
    eta_PR *= eta_PR_mult;
    eta_PS *= eta_PS_mult;
    eta_TA *= eta_TA_mult;
    eta_TL *= eta_TL_mult;

    // Find minimum spacing between tank macroaggregates by first finding the number of circles of tanks in hexagonal arrangement
    // that corresponds to the chosen TankNumber, and then converting it accounting for minimum intertank spacing
    // --------------------------------------------------------------------------------------------------------------------------
    // But waive mintankspacing if shape=12:
    if (shape==12) minTankSpacing = 0.;
    DefaultR2min    = D2min (TankNumber);
    Rslack2         = Rslack*Rslack;
    TankArea        = pow(TankRadius,2.)*pi*TankNumber;

    // Set fluxes of backgrounds
    // -------------------------
    fluxB_mu   = TankArea*Bgr_mu_per_m2;
    fluxB_e    = TankArea*Bgr_e_per_m2;

    // Define this here, as Ntrigger can have been redefined above
    // -----------------------------------------------------------
    SumProbRange = 2.*sqrt(1.*Ntrigger);

    // Override number of detectors for SWGO shapes
    // --------------------------------------------
    if (shape>100 && shape<115) {
        Nunits      = N_predef[shape-101];
        TankArea    = pow(TankRadius,2.)*pi;
        noSGDupdate = true;
    }

    // Safety checks
    // -------------
    cout    << endl;
    outfile << endl;
    if (PeVSource) {
        //if (!useN5s) {
        //    cout    << "     Warning, not using 5sigma, 3s derivative calc needs rechecking, switching to useN5s = true. " << endl;
        //    outfile << "     Warning, not using 5sigma, 3s derivative calc needs rechecking, switching to useN5s = true. " << endl;
        //    useN5s = true;
        //}
        if (E_PS<Emin) {
            cout    << "     PeV Source energy out of range - resetting E_PS to " << Emin*2. << endl;
            outfile << "     PeV Source energy out of range - resetting E_PS to " << Emin*2. << endl;
            E_PS = Emin*2.;
        }
        if (E_PS>Emax) {
            cout    << "     PeV Source energy out of range - resetting E_PS to " << 2./3.*Emax << endl;
            outfile << "     PeV Source energy out of range - resetting E_PS to " << 2./3.*Emax << endl;
            E_PS = Emax*2./3.;
        }
        if (SameShowers) {
            cout    << "     PeV Source is on - resetting SameShowers to false " << endl;
            outfile << "     PeV Source is on - resetting SameShowers to false " << endl;
            SameShowers = false;
        }
        // Redefine energy range for generation of showers
        // -----------------------------------------------
        Emin = E_PS*0.5;
        if (Emin<0.1) Emin = 0.1;
        Emax = E_PS*1.5;
        if (Emax>10.) Emax = 10.;
        if (Eslope!=0.) {
            cout    << "     PeV Source utility is on - resetting Eslope to 0." << endl;
            outfile << "     PeV Source utility is on - resetting Eslope to 0." << endl;
            Eslope = 0.;
        }
        // With PS true, it does not work if using true values in fit
        // ----------------------------------------------------------
        if (usetrueXY || usetrueE || usetrueAngs) {
            cout    << "     Sorry, when using a point source loss you need to fit for position, energy and angle." << endl;
            cout    << "     I am resetting to false the utx, ute, utp booleans. " << endl;
            outfile << "     Sorry, when using a point source loss you need to fit for position, energy and angle." << endl;
            outfile << "     I am resetting to false the utx, ute, utp booleans. " << endl;
            usetrueXY              = false;
            usetrueE               = false;
            usetrueAngs            = false;
        }
    }
    if (DynamicLR && !UseAreaCost && !UseLengthCost) {
        double sum = eta_GF + eta_IR + eta_PR;
        if (sum==eta_GF || sum==eta_IR || sum==eta_PR) {
            cout    << "     Warning, two LR are zero, turning off DynamicLR." << endl;
            outfile << "     Warning, two LR are zero, turning off DynamicLR." << endl;
            DynamicLR = false;
        }
    }
    if (fixE) {
        cout    << "     Warning, E is set to 1 PeV for all showers." << endl;
        outfile << "     Warning, E is set to 1 PeV for all showers." << endl;
    }
    if (setXYto00) {
        cout    << "     Warning, all showers xy are set to 0,0 " << endl;
        outfile << "     Warning, all showers xy are set to 0,0 " << endl;
    }
    if (StartLR<MinLearningRate) {
        cout    << "     Learning rate outside range [" << MinLearningRate << "," << MaxLearningRate << "]. Set to min" << endl;
        outfile << "     Learning rate outside range [" << MinLearningRate << "," << MaxLearningRate << "]. Set to min" << endl;
        StartLR = MinLearningRate;
    }
    if (StartLR>MaxLearningRate) {
        cout    << "     Learning rate outside range [" << MinLearningRate << "," << MaxLearningRate << "]. Set to max" << endl;
        outfile << "     Learning rate outside range [" << MinLearningRate << "," << MaxLearningRate << "]. Set to max" << endl;
        StartLR = MaxLearningRate;
    }
    if (Nunits<minUnits) {
        Nunits = 6;
        cout    << "     Too few units. Set to 6." << endl;
        outfile << "     Too few units. Set to 6." << endl;
    }
    if (Nunits*TankNumber<2.*Ntrigger) {
        Nunits = (int)(2.*Ntrigger/TankNumber) + 1;
        cout    << "     Too few total detector tanks given the chosen Ntrigger threshold. I reset Nunits to " << Nunits << endl;
        outfile << "     Too few total detector tanks given the chosen Ntrigger threshold. I reset Nunits to " << Nunits << endl;
    }
    if (Nunits>maxUnits) {
        cout    << "     Too many units. Stopping." << endl;
        outfile << "     Too many units. Stopping." << endl;
        return 0;
    }
    if (Nevents!=Nbatch) {
        cout    << "     Warning - you set Nevents and Nbatch to different values, this may create issues. Reset Nbatch = Nevents = " << Nevents << endl;
        outfile << "     Warning - you set Nevents and Nbatch to different values, this may create issues. Reset Nbatch = Nevents = " << Nevents << endl;
        Nbatch = Nevents;
    }
    if (Nevents+Nbatch>maxEvents) {
        cout    << "     Too many events. Resetting Nevents = Nbatch = 5000." << endl;
        outfile << "     Too many events. Resetting Nevents = Nbatch = 5000." << endl;
        Nevents = 5000;
        Nbatch  = 5000;
    }
    if (Nepochs>maxEpochs) {
        cout    << "     Too many epochs. Stopping." << endl;
        outfile << "     Too many epochs. Stopping." << endl;
        return 0;
    }
    if (DetectorSpacing<=0.) {
        cout    << "     DetectorSpacing must be >0. Stopping." << endl;
        outfile << "     DetectorSpacing must be >0. Stopping." << endl;
        return 0;
    }
    if (Eslope<-1. || Eslope>1.) {
        if (Eslope<-1.) Eslope = -1.;
        if (Eslope>1.)  Eslope = 1.;
        cout    << "     Sorry, too high |slope| in energy. Setting it to " << Eslope << endl;
        outfile << "     Sorry, too high |slope| in energy. Setting it to " << Eslope << endl;
    }
    if (Nsteps>maxNsteps) {
        cout    << "     Sorry max Nsteps = 500, set to that value " << endl;
        outfile << "     Sorry max Nsteps = 500, set to that value " << endl;
        Nsteps = 500;
    }
    if (DetectorSpacing*DetectorSpacing<DefaultR2min) {
        cout    << "     Sorry, you set the detector spacing too small given the required minimum spacing of tank aggregates. Reset to " << sqrt(DefaultR2min) << endl;
        outfile << "     Sorry, you set the detector spacing too small given the required minimum spacing of tank aggregates. Reset to " << sqrt(DefaultR2min) << endl;
        DetectorSpacing = sqrt(DefaultR2min);
    }
    if (SpacingStep*SpacingStep<DefaultR2min) {
        cout    << "     Sorry, you set the spacing step too small given the required minimum spacing of tank aggregates. Reset to " << sqrt(DefaultR2min) << endl;
        outfile << "     Sorry, you set the spacing step too small given the required minimum spacing of tank aggregates. Reset to " << sqrt(DefaultR2min) << endl;
        SpacingStep = sqrt(DefaultR2min);
    }
    //if (VoidRegion && CommonMode>0) {
    //    cout    << "     Warning, VoidRegion is on and CommonMode is >0. Are you sure you want to proceed? " << endl;
    //    outfile << "     Warning, VoidRegion is on and CommonMode is >0. Are you sure you want to proceed?" << endl;
    //}
    // Do not use the utility part connected to energy and pointing resolution if fixing E, angles to true values
    // ----------------------------------------------------------------------------------------------------------
    if (usetrueAngs) {
        eta_PR = 0.;
    }
    if (usetrueE) {
        eta_IR = 0.;
    }
    // if (CommonMode>0 && Nthreads>1) {
    //     cout << "     Sorry, cannot run with CommonMode>0 in multithreading mode. Stopping." << endl;
    //     return 0;
    // }

    bool MixedMode = false; // This bool determines if we have to use both CommonMode=0 and >1 because of some units losing coupling with others in a multiplet
    int Nmultiplets;
    if (CommonMode>=2) {
        if (shape<3 || (shape>4 && shape!=8 && shape!=9 && shape!=10)) {
            cout    << "     Sorry, cannot run with shape different from 3, 4, 8, 9, or 10 if CommonMode>=2. Stopping." << endl;
            outfile << "     Sorry, cannot run with shape different from 3, 4, 8, 9, or 10 if CommonMode>=2. Stopping." << endl;
            return 0;
        } else {
            if (Nunits%CommonMode!=0) {
                cout    << "     Sorry, with CommonMode>=2 we need to set Nunits multiple of CommonMode. Rounding it to " << Nunits+CommonMode-Nunits%CommonMode << endl; 
                outfile << "     Sorry, with CommonMode>=2 we need to set Nunits multiple of CommonMode. Rounding it to " << Nunits+CommonMode-Nunits%CommonMode << endl; 
               Nunits = Nunits + CommonMode - Nunits%CommonMode;
            }
        }
        multiplicity = CommonMode; // Otherwise it defaults at 1
        Nmultiplets = Nunits/multiplicity;
    }

    if (scanU) {
        if (!SameShowers) {
            cout    << "     With scanU you must set SameShowers true. I just did it for you." << endl;
            outfile << "     With scanU you must set SameShowers true. I just did it for you." << endl;
            SameShowers = true;
        }
        if (!fixShowerPos) {
            cout    << "     With scanU you must set fixShowerPos true. I will do that for you." << endl;
            outfile << "     With scanU you must set fixShowerPos true. I will do that for you." << endl;
            cout    << endl;
            outfile << endl;
            fixShowerPos = true;
        }
        cout << endl;
        cout    << "     Careful, scanU is set on, please ensure #FEWPLOTS is chosen when compiling." << endl; 
        outfile << "     Careful, scanU is set on, please ensure #FEWPLOTS is chosen when compiling." << endl; 
        cout << endl;
    }
    if (SameShowers) {
        if (Nevents%2!=0) Nevents++; // We need even Nevents in that case
        Nbatch = Nevents;
        cout    << "     SameShowers is on, fixed Nevents = Nbatch = " << Nevents << endl;
        outfile << "     SameShowers is on, fixed Nevents = Nbatch = " << Nevents << endl;
        if (Eslope!=0.) Eslope = 0.; 
        cout    << "     SameShowers is on, fixed Eslope = 0. " << endl;
        outfile << "     SameShowers is on, fixed Eslope = 0. " << endl;
        if (Nevents!=Nbatch) {
            cout    << "     Must have Nevents=Nbatch if SameShowers is true. Stopping." << endl;
            outfile << "     Must have Nevents=Nbatch if SameShowers is true. Stopping." << endl;
            return 0;
        }
        if (GenGammaFrac!=0.5) {
            cout    << "     Must have GenGammaFrac = 0.5 if SameShowers is true. Setting it so." << endl;
            outfile << "     Must have GenGammaFrac = 0.5 if SameShowers is true. Setting it so." << endl;
            GenGammaFrac = 0.5;
        }
        // if (!fixShowerPos && Nthreads>1) {
        //     cout    << "     Sorry, cannot run with SameShowers in multithreading mode. Stopping." << endl;
        //     outfile << "     Sorry, cannot run with SameShowers in multithreading mode. Stopping." << endl;
        //     return 0;
        // }
    }
    if (GenGammaFrac*Nbatch<50) {
        cout    << "     Warning! Too few gammas in batches. Are you sure? " << endl;
        outfile << "     Warning! Too few gammas in batches. Are you sure? " << endl;
    }
    if (OrthoShowers && SlantedShowers) {
        cout    << "     Sorry, cannot have both OrthoShowers and SlantedShowers on. Turning OrthoShowers off." << endl;
        outfile << "     Sorry, cannot have both OrthoShowers and SlantedShowers on. Turning OrthoShowers off." << endl;
        OrthoShowers = false;
    }
    if (usetrueE && eta_IR!=0.) {
        cout    << "     Using true energy, so I will set eta_IR = 0." << endl;
        outfile << "     Using true energy, so I will set eta_IR = 0." << endl;
        eta_IR = 0.;
    }
    if (usetrueAngs && eta_PR!=0.) {
        cout    << "     Using true angles, so I will set eta_PR = 0." << endl;
        outfile << "     Using true angles, so I will set eta_PR = 0." << endl;
        eta_PR = 0.;
    }

    // As long as Ntrigger is not large, we can initialize the factorials here
    // -----------------------------------------------------------------------
    if (Ntrigger>maxNtrigger) {
        cout    << "     Ntrigger is too large. Terminating. " << endl;
        outfile << "     Ntrigger is too large. Terminating. " << endl;
        return 0;
    } else {
        for (int i=0; i<maxNtrigger; i++) {
            F[i] = Factorial(i);
        }
    }
    if (Ntrigger>0.5*Nunits*TankNumber) {
        cout    << "     Sorry, Ntrigger > 0.5 Nunits, please change settings" << endl;
        outfile << "     Sorry, Ntrigger > 0.5 Nunits, please change settings" << endl;
        return 0;
    }

    // Other checks
    // ------------
    if (shape==3 && SpacingStep==0.) {
        cout    << "     Sorry, for circular shapes you need to define a radius increment larger than zero!" << endl;
        outfile << "     Sorry, for circular shapes you need to define a radius increment larger than zero!" << endl;
        return 0;
    }
    if (RelResCounts>0.2) { 
        cout    << "     Sorry, unsafe relative resolution, set to 0.2." << endl;
        outfile << "     Sorry, unsafe relative resolution, set to 0.2." << endl;
        RelResCounts = 0.2;
    }
    if (eta_GF==0 && eta_PS==0 && DisplFactor>1. && !scanU) {
        cout    << "     Warning. You are using a large step in position update with slowly varying IR and PR gradients. " << endl;
        outfile << "     Warning. You are using a large step in position update with slowly varying IR and PR gradients. " << endl;
        // DisplFactor = 1.;
    }

#if defined(STANDALONE) || defined(UBUNTU) 
    // Set up output file
    // ------------------
    string outPath  = GlobalPath + "Outputs/"; // "/lustre/cmswork/dorigo/swgo/MT/Outputs/";
    // Reserve Threads
    // ---------------
    threads.reserve(Nthreads);
#endif
#ifdef INROOT
    string outPath = "./SWGO/Outputs/";
#endif 

    // Determine first available file number to write.
    // Note: this indfile is static so it will be used also for the output detector geometry in SaveLayout(), 
    // so that the indices are aligned with one another. This is not the case of ReadLayout(), because an
    // output file corresponding to indfile might be missing. There, we take the last file before an empty slot.
    // ---------------------------------------------------------------------------------------------------------
    indfile = -1;
    ifstream tmpfile;
    char num[100];
    sprintf (num, "Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape);
     do {
        if (indfile>-1) tmpfile.close();
        indfile++;
        std::stringstream tmpstring;
        tmpstring << "RunDetails_" << num << "_Id=" << indfile;
        string tmpfilename = outPath + tmpstring.str() + ".txt";
        tmpfile.open(tmpfilename);
    } while (tmpfile.is_open());

    // Create the outfile for dump of event information
    // ------------------------------------------------
    std::stringstream sstr5; // This one includes the index
    sstr5 << "RunDetails_" << num << "_Id=" << indfile;
    string dump = outPath + sstr5.str() + ".txt";
    outfile.open(dump,ios::app); 

    outfile << endl;
    outfile << "     *****************************************************************" << endl;
    outfile << endl;
    outfile << "                          S   W   G   O   L   O                       " << endl;
    outfile << endl; 
    outfile << "         Southern Wide-field Gamma Observatory Layout Optimization    " << endl;
    outfile << endl;
    outfile << "         Proof-of-principle study                                     " << endl;
    outfile << "         of SWGO detector optimization with end-to-end model          " << endl;
    outfile << endl;
    outfile << "                        T. Dorigo, M. Doro, L. Recabarren 2022-2024   " << endl;
    outfile << endl;
    outfile << "     *****************************************************************" << endl;
    outfile << endl;
    outfile << "     Running with the following parameters: " << endl;
    outfile << "     Nevents         = " << Nevents << endl;
    outfile << "     Nbatch          = " << Nbatch << endl;
    outfile << "     Nunits          = " << Nunits << endl;
    outfile << "     Nepochs         = " << Nepochs << endl;
    outfile << "     DetectorSpacing = " << DetectorSpacing << endl;
    outfile << "     SpacingStep     = " << SpacingStep << endl;
    outfile << "     DisplFactor     = " << DisplFactor << endl;
    outfile << "     Rslack          = " << Rslack << endl;
    outfile << "     shape           = " << shape << endl;
    outfile << "     CommonMode      = " << CommonMode << endl;
    outfile << "     StartLR         = " << StartLR << endl;
    outfile << "     Ngrid           = " << Ngrid << endl;
    outfile << "     NEgrid          = " << NEgrid << endl;
    outfile << "     Nsteps          = " << Nsteps << endl;
    outfile << "     LRX             = " << LRX << endl;
    outfile << "     LRE             = " << LRE << endl;
    outfile << "     LRA             = " << LRA << endl;
    outfile << "     Eslope          = " << Eslope << endl;
    outfile << "     Nthreads        = " << Nthreads << endl;
    outfile << "     eta_GF_mult     = " << eta_GF_mult << endl;
    outfile << "     eta_IR_mult     = " << eta_IR_mult << endl;
    outfile << "     eta_PR_mult     = " << eta_PR_mult << endl;
    outfile << "     eta_PS_mult     = " << eta_PS_mult << endl;
    outfile << "     eta_TA_mult     = " << eta_TA_mult << endl;
    outfile << "     eta_TL_mult     = " << eta_TL_mult << endl;
    outfile << "     startEpoch      = " << startEpoch << endl;
    outfile << "     Ntrigger        = " << Ntrigger << endl;
#ifdef EXPANDARRAY
    outfile << "     TankNumber      = 1 " << endl;
#else
    outfile << "     TankNumber      = " << TankNumber << endl;
#endif
    outfile << "     addSysts        = " << addSysts << endl;
    outfile << "     RelResCounts    = " << RelResCounts << endl;
    outfile << "     usetrueXY       = " << usetrueXY << endl;
    outfile << "     usetrueE        = " << usetrueE << endl;
    outfile << "     usetrueAngs     = " << usetrueAngs << endl;
    outfile << "     OrthoShowers    = " << OrthoShowers << endl;
    outfile << "     SlantedShowers  = " << SlantedShowers << endl;
    outfile << "     scanU           = " << scanU << endl;
    outfile << "     readGeom        = " << readGeom << endl;
    outfile << "     noSGDupdate     = " << noSGDupdate << endl;
    outfile << "     initTrueVals    = " << initTrueVals << endl;
    outfile << "     initBitmap      = " << initBitmap << endl;
    outfile << "     fixShowerPos    = " << fixShowerPos << endl;
    outfile << "     HexaShowers     = " << HexaShowers << endl;
    outfile << "     SameShowers     = " << SameShowers << endl;
    outfile << "     idstar          = " << idstar << endl;
    outfile << "     writeGeom       = " << writeGeom << endl;
    outfile << "     Rmin            = " << Rmin << endl;
    outfile << "     TankArea/pi     = " << TankArea/pi << endl;
    outfile << "     Wslope          = " << Wslope << endl;
    outfile << "     sigma_time      = " << sigma_time << endl;
    outfile << "     sigma_texp      = " << sigma_texp << endl;
    outfile << "     logLApprox      = " << logLApprox << endl;
    outfile << "     Beta1           = " << beta1 << endl;
    outfile << "     Beta2           = " << beta2 << endl;
    outfile << "     PlotBitmap      = " << plotBitmap << endl;
    outfile << "     UseAreaCost     = " << UseAreaCost << endl;
    outfile << "     UseLengthCost   = " << UseLengthCost << endl;
    outfile << "     PeVSource       = " << PeVSource << endl;
    outfile << "     E_PS            = " << E_PS << endl;
    outfile << "     VoidRegion      = " << VoidRegion << endl;
    outfile << "     useN5s          = " << useN5s << endl;
    outfile << endl;
    outfile << "     *****************************************************************" << endl;

    cout    << endl;
    cout    << endl;
    cout    << "     *****************************************************************" << endl;
    cout    << endl;
    cout    << "                          S   W   G   O   L   O                       " << endl;
    cout    << endl; 
    cout    << "         Southern Wide-field Gamma Observatory Layout Optimization    " << endl;
    cout    << endl;
    cout    << "         Proof-of-principle study                                     " << endl;
    cout    << "         of SWGO detector optimization with end-to-end model          " << endl;
    cout    << endl;
    cout    << "                       T. Dorigo, M. Doro, L. Recabarren 2022-2024    " << endl;
    cout    << endl;
    cout    << "     *****************************************************************" << endl;
    cout    << endl;
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
    gStyle->SetPalette(55); // kRainBow in root6
    gStyle->SetNumberContours(99);
 
    // Get a sound RN generator
    // ------------------------
    delete gRandom;
    myRNG  = new TRandom3();
    myRNG2 = new TRandom3();

    // If RndmSeed is on, we get a different simulation every time
    // -----------------------------------------------------------
    if (RndmSeed) {
        myRNG->SetSeed(static_cast<UInt_t>(std::time(0)));
        myRNG2->SetSeed(static_cast<UInt_t>(std::time(0)));
    } else {
        myRNG->SetSeed(0.);
        myRNG2->SetSeed(0.);
    }

    // Suppress root warnings
    // ----------------------
    gROOT->ProcessLine ("gErrorIgnoreLevel = 6001;");
    gROOT->ProcessLine ("gPrintViaErrorHandler = kTRUE;");
    //gStyle->SetPalette(5);

    // Setup the output root file path
    // -------------------------------
#if defined(STANDALONE) || defined(UBUNTU) 
    string rootPath = GlobalPath + "Root/"; // "/lustre/cmswork/dorigo/swgo/MT/Root/";
#endif
#ifdef INROOT
    string rootPath = "./SWGO/Root/";
#endif
     
    // Define the current geometry 
    // ---------------------------
    if (readGeom) {
        ReadLayout ();
    } else {
        DefineLayout (); // Also defines initial ArrayRspan
    }

    // Define initial positions (used to fill Rdistr0 histogram while varying ArrayRspan)
    // -----------------------------------------------------------------------------
    for (int id=0; id<Nunits; id++) {
        xinit[id] = x[id];
        yinit[id] = y[id];
        // Also initialize xprev, yprev since we are at it (otherwise they may not get initialized in the sgd loop if scanU is true)
        // -------------------------------------------------------------------------------------------------------------------------
        xprev[id] = x[id];
        yprev[id] = y[id];
    }

    // Fill vector of Gaussian shifts for histogramming of LLR PDFs
    // ------------------------------------------------------------
    for (int i=0; i<10000; i++) {
        shift[i] = myRNG->Gaus(0.,1.);
    }
    NumAvgSteps = 0;
    DenAvgSteps = 0;

    // Define number of R bins depending on initial value of ArrayRspan (which is now defined) and Rslack
    // ---------------------------------------------------------------------------------------------
    // NRbins = 10*TotalRspan/ArrayRspan[0];
    // if (NRbins>maxRbins) NRbins = maxRbins;
    NRbins = 5*log(Nunits)+1;

    TrueGammaFraction = 0.5; // But some events may end up becoming inactive, see below.
    
    // Read in parametrizations of particle fluxes and lookup table
    // ------------------------------------------------------------
    int code = ReadShowers ();
    if (code!=0) {
        cout    << "     Unsuccessful retrieval of shower parameters from ascii files, terminating." << endl;
        outfile << "     Unsuccessful retrieval of shower parameters from ascii files, terminating." << endl;
        outfile.close();
        return 0;
    }

    // Compute the correction to the flux estimates, as a function of the distance of the tank from the core
    // -----------------------------------------------------------------------------------------------------
    if (StudyFluxRatio) {
        TH1D * RatioEG[100];
        TH1D * RatioMG[100];
        TH1D * RatioEP[100];
        TH1D * RatioMP[100];
        char namer[50];
        for (int i=0; i<100; i++) {
            sprintf (namer,"RatioEG%d",i);
            RatioEG[i] = new TH1D (namer, namer, 100, 0., 10.); // 0 to 10 meters
            sprintf (namer,"RatioMG%d",i);
            RatioMG[i] = new TH1D (namer, namer, 100, 0., 10.); // 0 to 10 meters
            sprintf (namer,"RatioEP%d",i);
            RatioEP[i] = new TH1D (namer, namer, 100, 0., 10.); // 0 to 10 meters
            sprintf (namer,"RatioMP%d",i);
            RatioMP[i] = new TH1D (namer, namer, 100, 0., 10.); // 0 to 10 meters
        }
        for (int ie=0; ie<10; ie++) {
            double energy = 0.5+ie*1.;
            for (int it=0; it<10; it++) {
                double theta = thetamax/10.*(it+0.5);
                for (int idist = 0; idist<100; idist++) {
                    double R = 0.1*idist; // We compute this from 0 to 10 meters distance from the center of the tank
                    double sumfluxEG = 0.;
                    double sumfluxMG = 0.;
                    double sumfluxEP = 0.;
                    double sumfluxMP = 0.;
                    double sumden  = 0.;
                    int Nphi = 36;
                    int Nr   = 50;
                    for (int iphi=0; iphi<Nphi; iphi++) {
                        double cosphi = cos((iphi+0.5)*twopi/Nphi);
                        for (int ir=0; ir<Nr; ir++) {
                            double r = (ir+0.5)*TankRadius/Nr;
                            double d = sqrt(R*R+r*r-2.*r*R*cosphi); // Cosine rule to get distance of core to point (r,phi) on the circle
                            sumfluxEG += r*EFromG (energy, theta, d, 0);
                            sumfluxMG += r*MFromG (energy, theta, d, 0);
                            sumfluxEP += r*EFromP (energy, theta, d, 0);
                            sumfluxMP += r*MFromP (energy, theta, d, 0);
                            sumden  += r;
                        }
                    }
                    double flux_at_center_EG = EFromG (energy,theta,R,0);
                    double flux_at_center_MG = MFromG (energy,theta,R,0);
                    double flux_at_center_EP = EFromP (energy,theta,R,0);
                    double flux_at_center_MP = MFromP (energy,theta,R,0);
                    double ratioEG = sumfluxEG/sumden/flux_at_center_EG;
                    double ratioMG = sumfluxMG/sumden/flux_at_center_MG;
                    double ratioEP = sumfluxEP/sumden/flux_at_center_EP;
                    double ratioMP = sumfluxMP/sumden/flux_at_center_MP;
                    RatioEG[ie*10+it]->SetBinContent(idist+1,ratioEG);
                    RatioMG[ie*10+it]->SetBinContent(idist+1,ratioMG);
                    RatioEP[ie*10+it]->SetBinContent(idist+1,ratioEP);
                    RatioMP[ie*10+it]->SetBinContent(idist+1,ratioMP);
                }
            }
        }
        TCanvas * PR = new TCanvas ("PR","", 900, 900);
        PR->Divide(2,2);
        PR->cd(1);
        RatioEG[0]->Draw();
        PR->cd(2);
        RatioMG[0]->Draw();
        PR->cd(3);
        RatioEP[0]->Draw();
        PR->cd(4);
        RatioMP[0]->Draw();
        for (int i=1; i<100; i++) {
            PR->cd(1);
            RatioEG[i]->Draw("SAME");
            PR->cd(2);
            RatioMG[i]->Draw("SAME");
            PR->cd(3);
            RatioEP[i]->Draw("SAME");
            PR->cd(4);
            RatioMP[i]->Draw("SAME");
        }

        // Write these histograms to a special file
        // ----------------------------------------
        std::stringstream rootstr0;
        rootstr0 << "Swgolo141_fluxes";
        string namerootfile0 = rootPath  + rootstr0.str() + ".root";
        TFile * rootfile0 = new TFile (namerootfile0.c_str(),"RECREATE");
        rootfile0->cd();
        PR->Write();
        for (int i=0; i<100; i++) {
            RatioEG[i]->Write();
            RatioMG[i]->Write();
            RatioEP[i]->Write();
            RatioMP[i]->Write();
        }
        rootfile0->Close();
    } // end if StudyFluxRatio 
    
    // Decompose plotting bitmap into an array
    // ---------------------------------------
    // PlotThis[0]  -> U, Uave                  - use by default, +1
    // PlotThis[1]  -> Layout + Showers3 / 3p   - use by default, +2
    // PlotThis[2]  -> Rdistr0, Rdistr, DR?     - use by default, +4
    // PlotThis[3]  -> UaveGF, IR, PR, TC       - use by default, +8
    // PlotThis[4]  -> DR, DR0                  - use by default, +16 if not useTrueAngs
    // PlotThis[5]  -> DE, DE0                  - use by default, +32 if not useTrueE
    // PlotThis[6]  -> LLRG, LLRP (or Pareto)   - +64
    // PlotThis[7]  -> DUGF, DUIR, DUPR, DUTC   - +128
    // PlotThis[8]  -> DR0vsE                   - +256
    // PlotThis[9]  -> HEtrue, HEmeas           - use by default, +512
    // PlotThis[10] -> CosDir                   - +1024
    // PlotThis[11] -> CosvsEp                  - +2048
    // PlotThis[12] -> Layout2                  - use by default, +4096
    // PlotThis[13] -> ThGIvsThGP               - +8192
    // PlotThis[14] -> LrGivsLrGP               - +16384
    // PlotThis[15] -> DUDR                     - +32768
    //
    // Useful codes: 4671 (8 default plots), 5135, 14271 
    // -------------------------------------------------
    
    int plotBitmap_curr = plotBitmap;
    int Nplots_CT = 0;
    for (int i=15; i>=0; i--) {
        PlotThis[i] = false;
        int pow2 = (int)pow(2.,i);
        if (plotBitmap_curr>=pow2) {
            plotBitmap_curr-=pow2;
            PlotThis[i] = true;
            Nplots_CT++;
        } 
    }
    // Reset PlotThis array values for specific running conditions
    // -----------------------------------------------------------
    if (noSGDupdate) {
        if (PlotThis[0]) {
            PlotThis[0] = false;
            Nplots_CT--;
        }
        if (PlotThis[3]) { // 4,5,7,10,11,13,14,15
            PlotThis[3] = false;
            Nplots_CT--;
        }
        if (PlotThis[4]) { // 4,5,7,10,11,13,14,15
            PlotThis[4] = false;
            Nplots_CT--;
        }
        if (PlotThis[5]) { // 4,5,7,10,11,13,14,15
            PlotThis[5] = false;
            Nplots_CT--;
        }
        if (PlotThis[7]) { // 4,5,7,10,11,13,14,15
            PlotThis[7] = false;
            Nplots_CT--;
        }
        if (PlotThis[10]) { // 4,5,7,10,11,13,14,15
            PlotThis[10] = false;
            Nplots_CT--;
        }
        if (PlotThis[11]) { // 4,5,7,10,11,13,14,15
            PlotThis[11] = false;
            Nplots_CT--;
        }
        if (PlotThis[13]) { // 4,5,7,10,11,13,14,15
            PlotThis[13] = false;
            Nplots_CT--;
        }
        if (PlotThis[14]) { // 4,5,7,10,11,13,14,15
            PlotThis[14] = false;
            Nplots_CT--;
        }
        if (PlotThis[15]) { // 4,5,7,10,11,13,14,15
            PlotThis[15] = false;
            Nplots_CT--;
        }
    } // end if noSGDUpdate
    if (usetrueAngs) {
        if (PlotThis[4]) { 
            PlotThis[4] = false;
            Nplots_CT--;
        }
    }
    if (usetrueE) {
        if (PlotThis[5]) { 
            PlotThis[5] = false;
            Nplots_CT--;
        }
    }

#ifdef PLOTS
    // Define canvases for check of model
    // ----------------------------------
#endif
    if (checkmodel) { 
        TCanvas * TMPe0  = new TCanvas ("TMPe0","", 800,800); 
        TCanvas * TMPe1  = new TCanvas ("TMPe1","", 800,800); 
        TCanvas * TMPe2  = new TCanvas ("TMPe2","", 800,800); 
        TCanvas * TMPe3  = new TCanvas ("TMPe3","", 800,800); 
        TCanvas * TMPm0  = new TCanvas ("TMPm0","", 800,800); 
        TCanvas * TMPm1  = new TCanvas ("TMPm1","", 800,800); 
        TCanvas * TMPm2  = new TCanvas ("TMPm2","", 800,800); 
        TCanvas * TMPm3  = new TCanvas ("TMPm3","", 800,800); 
        TCanvas * mgflux = new TCanvas ("mgflux","",500,500);
        TCanvas * mpflux = new TCanvas ("mpflux","",500,500);
        TCanvas * egflux = new TCanvas ("egflux","",500,500);
        TCanvas * epflux = new TCanvas ("epflux","",500,500);
        TH1D * Rg_e[80];
        TH1D * Rg_m[80];
        TH1D * Rp_e[80];
        TH1D * Rp_m[80];
        char hname[50];
        for (int i=0; i<80; i++) {
            sprintf (hname, "Rg_m%d",i);
            Rg_m[i] = new TH1D (hname,hname,500, 0., 2500.); 
            sprintf (hname, "Rg_e%d",i);
            Rg_e[i] = new TH1D (hname,hname,500, 0., 2500.); 
            sprintf (hname, "Rp_m%d",i);
            Rp_m[i] = new TH1D (hname,hname,500, 0., 2500.); 
            sprintf (hname, "Rp_e%d",i);
            Rp_e[i] = new TH1D (hname,hname,500, 0., 2500.); 
            Rp_e[i]->SetLineColor(kRed);
            Rp_m[i]->SetLineColor(kRed);
        }
        // Fill histograms of radial densities
        // -----------------------------------
        for (int ie=0; ie<20; ie++) {
            double e = exp(logdif*(ie+0.5)/20. + log_01); 
            for (int it=0; it<4; it++) {
                // 0.5 = 0., 4.5 = thetamax. We get values at 1,2,3,4
                double t = thetamax/4.*(it+0.5);
                for (int i=0; i<500; i++) {
                    double r = 2.5+5*i;
                    Rg_m[ie*4+it]->SetBinContent (i+1,MFromG(e,t,r,0)/TankArea);
                    Rg_e[ie*4+it]->SetBinContent (i+1,EFromG(e,t,r,0)/TankArea);
                    Rp_m[ie*4+it]->SetBinContent (i+1,MFromP(e,t,r,0)/TankArea);
                    Rp_e[ie*4+it]->SetBinContent (i+1,EFromP(e,t,r,0)/TankArea);
                    if (it==0) {
                        if (i%400==0) cout << "mg energy = " << e << " theta = " << t << " R = " << i+0.5 << " " << MFromG(e,t,r,0)/TankArea << endl;
                        if (i%400==0) cout << "eg energy = " << e << " theta = " << t << " R = " << i+0.5 << " " << EFromG(e,t,r,0)/TankArea << endl;
                        if (i%400==0) cout << "mp energy = " << e << " theta = " << t << " R = " << i+0.5 << " " << MFromP(e,t,r,0)/TankArea << endl;
                        if (i%400==0) cout << "ep energy = " << e << " theta = " << t << " R = " << i+0.5 << " " << EFromP(e,t,r,0)/TankArea << endl;
                    }
                }
            }   
        }

#ifdef PLOTS
        // Plot them
        // ---------
        TMPe0->Divide(4,5);
        TMPe1->Divide(4,5);
        TMPe2->Divide(4,5);
        TMPe3->Divide(4,5);
        TMPm0->Divide(4,5);
        TMPm1->Divide(4,5);
        TMPm2->Divide(4,5);
        TMPm3->Divide(4,5);
        for (int i=1; i<=20; i++) {
            TMPe0->cd(i);
            TMPe0->GetPad(i)->SetLogy();
            Rg_e[i*4-4]->Draw();
            Rp_e[i*4-4]->Draw("SAME");
        }
        for (int i=1; i<=20; i++) {
            TMPe1->cd(i);
            TMPe1->GetPad(i)->SetLogy();
            Rg_e[i*4-3]->Draw();
            Rp_e[i*4-3]->Draw("SAME");
        }
        for (int i=1; i<=20; i++) {
            TMPe2->cd(i);
            TMPe2->GetPad(i)->SetLogy();
            Rg_e[i*4-2]->Draw();
            Rp_e[i*4-2]->Draw("SAME");
        }
        for (int i=1; i<=20; i++) {
            TMPe3->cd(i);
            TMPe3->GetPad(i)->SetLogy();
            Rg_e[i*4-1]->Draw();
            Rp_e[i*4-1]->Draw("SAME");
        }
        for (int i=1; i<=20; i++) {
            TMPm0->cd(i);
            TMPm0->GetPad(i)->SetLogy();
            Rg_m[i*4-4]->Draw();
            Rp_m[i*4-4]->Draw("SAME");
        }
        for (int i=1; i<=20; i++) {
            TMPm1->cd(i);
            TMPm1->GetPad(i)->SetLogy();
            Rg_m[i*4-3]->Draw();
            Rp_m[i*4-3]->Draw("SAME");
        }
        for (int i=1; i<=20; i++) {
            TMPm2->cd(i);
            TMPm2->GetPad(i)->SetLogy();
            Rg_m[i*4-2]->Draw();
            Rp_m[i*4-2]->Draw("SAME");
        }
        for (int i=1; i<=20; i++) {
            TMPm3->cd(i);
            TMPm3->GetPad(i)->SetLogy();
            Rg_m[i*4-1]->Draw();
            Rp_m[i*4-1]->Draw("SAME");
        }
        // Plots of parameters versus E and theta
        // --------------------------------------
        mgflux->Divide(2,1);
        mgflux->cd(1);
        P0mg->Draw("COLZ");
        mgflux->cd(2);
        P2mg->Draw("COLZ");
        mpflux->Divide(2,1);
        mpflux->cd(1);
        P0mp->Draw("COLZ");
        mpflux->cd(2);
        P2mp->Draw("COLZ");
        egflux->Divide(3,1);
        egflux->cd(1);
        P0eg->Draw("COLZ");
        egflux->cd(2);
        P1eg->Draw("COLZ");
        egflux->cd(3);
        P2eg->Draw("COLZ");
        epflux->Divide(3,1);
        epflux->cd(1);
        P0ep->Draw("COLZ");
        epflux->cd(2);
        P1ep->Draw("COLZ");
        epflux->cd(3);
        P2ep->Draw("COLZ");
#endif
    } // End if check model 

    // Check that everything is in order
    // ---------------------------------
    if (warnings1+warnings2+warnings3!=0) {
        TerminateAbnormally();
        return 0;
    }

    // Big optimization loop, modifying detector layout
    // ------------------------------------------------
    outfile << "     Starting gradient descent loop. "; 
    if (!UseAreaCost) outfile << endl << endl;
    cout    << "     Starting gradient descent loop. "; 
    if (!UseAreaCost) cout << endl << endl;
    double maxUtility = 0.;
    int imax = 0;

    // Einit for FitShowerParams routine
    // ---------------------------------
    for (int ie=0; ie<NEgrid; ie++) {
        Einit[ie] = Emin*pow(pow(Emax/Emin,1./NEgrid),ie);
    }

    // Beta values for ADAM grad descent
    // ---------------------------------
    for (int ist=0; ist<Nsteps+1; ist++) {
        powbeta1[ist] = pow(beta1,ist);
        powbeta2[ist] = pow(beta2,ist);
        if (powbeta1[ist]<1./largenumber) powbeta1[ist] = 1./largenumber;
        if (powbeta2[ist]<1./largenumber) powbeta2[ist] = 1./largenumber;
    }

    // Histogram definition
    // --------------------
    double rangex = DetectorSpacing*6.;  // Used if scanU is true 
    double rangey = DetectorSpacing*6.;  // Used if scanU is true
    int NbinsRdistr  = Nunits/10;
    if (NbinsRdistr<20) NbinsRdistr = 20;
    int NbinsPdistr  = Nunits/30;
    if (NbinsPdistr<3) NbinsPdistr = 3;
    int NbinsProfU;
    if (Nepochs<5) {
        NbinsProfU = 1;
    } else if (Nepochs<1000) {
        NbinsProfU = Nepochs/5;
    } else {
        NbinsProfU = 200.;
    }
    int NepPerBin=Nepochs/NbinsProfU;
    TH1D * U           = new TH1D     ("U",          "Utility function versus epoch",  Nepochs,    -0.5, (double)Nepochs-0.5);  
    TProfile * Uave    = new TProfile ("Uave",       "Utility ratios for components",  NbinsProfU, -0.5, (double)Nepochs-0.5, 0., 100000.);  
    TProfile * UaveGF  = new TProfile ("UaveGF",     "Average flux utility",           NbinsProfU, -0.5, (double)Nepochs-0.5, 0., 100000.);  
    TProfile * UaveIR  = new TProfile ("UaveIR",     "Average energy res. utility",    NbinsProfU, -0.5, (double)Nepochs-0.5, 0., 100000.);  
    TProfile * UavePR  = new TProfile ("UavePR",     "Average pointing res. utility",  NbinsProfU, -0.5, (double)Nepochs-0.5, 0., 100000.);  
    TProfile * UaveTC  = new TProfile ("UaveTC",     "Average cost utility",           NbinsProfU, -0.5, (double)Nepochs-0.5, 0., 100000.);  
    TH1D * CosDir      = new TH1D     ("CosDir",     "Alignment of successive moves",  20, -1., 1.); // 20, 0., pi);
    TProfile * CosvsEp = new TProfile ("CosvsEp",    "Alignment versus epoch",         Nepochs, -0.5, Nepochs-0.5, -1., 1.); // 0., pi);
    TH1D * HEtrue      = new TH1D     ("HEtrue",     "True E spectrum",                10, 0., 10.);
    TH1D * HEmeas      = new TH1D     ("HEmeas",     "Measured E spectrum",            10, 0., 10.);
    TProfile * dUdR    = new TProfile ("dUdR",       "Average dU/dR vs epoch",         NbinsProfU, -0.5, (double)Nepochs-0.5, -500., 500.);
    TH2D * Pareto      = new TH2D     ("Pareto",     "Pareto front UPR vs UIR",        200, 0., 1000., 200, 500., 1500.);

    TH2D * Layout    = nullptr;  
    TH2D * Layout2   = nullptr;  
    bool plotdensity = true; // This defines whether we plot the errors on position or energy estimates as a color map of Showers3 scatterplot
    TH2D * Showers3  = nullptr;
    //TImage * Pampalabola = nullptr; 
    TLine * Line1    = nullptr;
    TLine * Line2    = nullptr;
    TLine * Line3    = nullptr;
    TProfile2D * Showers3p = nullptr;
    TH1D * Rdistr0   = nullptr;  // Changes at every sgd loop 
    TH1D * Rdistr    = nullptr;
    TH1D * Pdistr0   = nullptr;  // Changes at every sgd loop 
    TH1D * Pdistr    = nullptr;

#ifdef FEWPLOTS 
    TH1D * LLRP      = nullptr;
    TH1D * LLRG      = nullptr;

    // Histograms filled if scanU is true
    // ----------------------------------
    TH2D * Uvsxy       = new TH2D     ("Uvsxy",    "", (int)(sqrt((double)Nepochs)),x[idstar]-rangex,x[idstar]+rangex,                                                     
                                                       (int)(sqrt((double)Nepochs)),y[idstar]-rangey,y[idstar]+rangey);
    TH1D * Uvsx        = new TH1D     ("Uvsx",     "", (int)(sqrt((double)Nepochs)),x[idstar]-rangex,x[idstar]+rangex);                                                    
    TH1D * Uvsy        = new TH1D     ("Uvsy",     "", (int)(sqrt((double)Nepochs)),y[idstar]-rangey,y[idstar]+rangey);                                                     
#endif

#ifdef PLOTS
    TH1D * LLRP = nullptr;
    TH1D * LLRG = nullptr;
    TH1D * GFmeas      = new TH1D     ("GFmeas",   "", 5, -3.5, 1.5);
    TH1D * GFtrue      = new TH1D     ("GFtrue",   "", 5, -3.5, 1.5);
    TProfile * dUdx    = new TProfile ("dUdx",     "Derivative dU/dx", Nepochs, -0.5, Nepochs-0.5, 0., 100.);
    TH1D * PosQ        = new TH1D     ("PosQ",     "Quality of position estimate", Nepochs, -0.5, Nepochs-0.5);
    TH1D * AngQ        = new TH1D     ("AngQ",     "Quality of angular estimate", Nepochs, -0.5, Nepochs-0.5);
    TH1D * EQ          = new TH1D     ("EQ",       "Quality of energy estimate", Nepochs, -0.5, Nepochs-0.5);
    TH2D * LR          = new TH2D     ("LR",       "", Nepochs, -0.5, Nepochs-0.5, 100, -5., 5.);
    TH1D * U_gf        = new TH1D     ("U_gf",     "Flux-dependent Utility contribution", Nepochs, -0.5, Nepochs-0.5);
    TH1D * U_pr        = new TH1D     ("U_pr",     "Position resolution Utility", Nepochs, -0.5, Nepochs-0.5);
    TH1D * U_ir        = new TH1D     ("U_ir",     "Energy resolution Utility", Nepochs, -0.5, Nepochs-0.5);
    TH1D * U_tc        = new TH1D     ("U_tc",     "Total cost utility (area, length)", Nepochs, -0.5, Nepochs-0.5);
    TH1D * PG          = new TH1D     ("PG", "PDF of test statistic for gammas",640,-32.,0.);
    TH1D * PP          = new TH1D     ("PP", "PDF of test statistic for protons",640,-32.,0.);

    // These are filled in FitShowerParams so already declared static
    // ---------------------------------------------------------------
    NumStepsg          = new TProfile ("NumStepsg",  "", 20, 0.1, 10., 0., Nsteps+1); // To study logLApprox
    NumStepsp          = new TProfile ("NumStepsp",  "", 20, 0.1, 10., 0., Nsteps+1); // To study logLApprox
#endif

    Pareto->SetMarkerStyle(20);
    Pareto->SetMarkerSize(0.4);
    U->SetMarkerStyle(20);
    U->SetMarkerSize(0.4);
    U->SetMinimum(0.);
    Uave->SetLineColor(kRed);
    UaveGF->SetLineColor(kBlue);
    UaveIR->SetLineColor(kBlack);
    UavePR->SetLineColor(kRed);
    UaveTC->SetLineColor(kGreen);
    UaveGF->SetLineWidth(3);
    UaveIR->SetLineWidth(3);
    UavePR->SetLineWidth(3);
    UaveTC->SetLineWidth(3);
    UaveGF->Fill(1,1.);
    UaveGF->Fill(1,0.99);
    UaveGF->Fill(1,1.01);
    UaveIR->Fill(1,1.);
    UaveIR->Fill(1,0.99);
    UaveIR->Fill(1,1.01);
    UavePR->Fill(1,1.);
    UavePR->Fill(1,0.99);
    UavePR->Fill(1,1.01);
    UaveTC->Fill(1,1.);
    UaveTC->Fill(1,0.99);
    UaveTC->Fill(1,1.01);
    //UaveGF->SetBinContent(1,1.);
    //UaveIR->SetBinContent(1,1.);
    //UavePR->SetBinContent(1,1.);
    //UaveGF->SetBinError(1,0.01); // Hack - undefined for first bin
    //UaveIR->SetBinError(1,0.01);
    //UavePR->SetBinError(1,0.01);
    ThGIvsThGP->SetMarkerStyle(20);
    ThGIvsThGP->SetMarkerSize(0.4);
    LrGIvsLrGP->SetMarkerStyle(20);
    LrGIvsLrGP->SetMarkerSize(0.4);
    DE0->SetLineColor(kRed);
    DE0->SetLineWidth(3);
    DE->SetLineColor(kBlue);
    DE->SetLineWidth(3);
    DR0->SetLineColor(kRed);
    DR0->SetLineWidth(3);
    double dRMin  = -7.; // We use these to reset the vertical scale of the graph. Defined here to avoid recomputing them for DR0 in loop
    double dRMax  = -5.;
    DR->SetLineColor(kBlue);
    DR->SetLineWidth(3);
    //numx->SetLineColor(kRed);
    //denx->SetLineColor(kRed);
    //dxrec->SetLineColor(kRed);
#ifdef PLOTRESOLUTIONS
    DE0vsE->SetLineColor(kRed);
    DE0vsE->SetLineWidth(3);
    DE0vsR->SetLineColor(kBlue);
    DE0vsR->SetLineWidth(3);
    DR0vsE->SetLineColor(kRed);
    DR0vsR->SetLineColor(kBlue);
    DR0vsR->SetLineWidth(3);
    DR0vsE->SetLineWidth(3);
#endif
    DUGF->SetLineWidth(3);
    DUIR->SetLineWidth(3);
    DUPR->SetLineWidth(3);
    DUTC->SetLineWidth(3);
    DUGF->SetLineColor(kBlue);
    DUIR->SetLineColor(kBlack);
    DUPR->SetLineColor(kRed);
    DUTC->SetLineColor(kGreen);
    HEtrue->SetLineColor(kRed);
    HEtrue->SetMinimum(0);
    HEtrue->SetLineWidth(3);
    HEmeas->SetLineWidth(3);

    char namededr[50];
    for (int i=0; i<10; i++) {
        sprintf (namededr,"dEdR%d",i);
        dEdR[i] = new TH1D (namededr, namededr,100,0.,DetectorSpacing);
    }

#ifdef PLOTS
    EQ->SetLineColor(kRed);
    EQ->SetLineWidth(3);
    U_ir->SetLineWidth(3);
    U_ir->SetLineColor(kRed);
    U_gf->SetLineWidth(3);
    PosQ->SetLineWidth(3);
    AngQ->SetLineWidth(3);
    NumStepsp->SetLineColor(kRed);
    PP->SetLineColor(kRed);

    // Define histograms for RMS calculations
    // --------------------------------------
    char namer[50];
    for (int ib=0; ib<NbinsResR; ib++) {
        for (int jb=0; jb<NbinsResE; jb++) {
            int ih = ib*NbinsResE+jb;
            sprintf (namer, "DXYg%d",ih);
            DXYg[ih] = new TH1D (namer, namer, 100,0.,maxdxy);
            sprintf (namer, "DXYp%d",ih);
            DXYp[ih] = new TH1D (namer, namer, 100,0.,maxdxy);
            sprintf (namer, "DE_g%d",ih);
            DE_g[ih] = new TH1D (namer, namer, 100,0.,10.);
            sprintf (namer, "DE_p%d",ih);
            DE_p[ih] = new TH1D (namer, namer, 100,0.,10.);
            sprintf (namer, "DT_g%d",ih);
            DT_g[ih] = new TH1D (namer, namer, 100,0.,pi/2.);
            sprintf (namer, "DT_p%d",ih);
            DT_p[ih] = new TH1D (namer, namer, 100,0.,pi/2.);
            sprintf (namer, "DP_g%d",ih);
            DP_g[ih] = new TH1D (namer, namer, 100,0.,pi);
            sprintf (namer, "DP_p%d",ih);
            DP_p[ih] = new TH1D (namer, namer, 100,0.,pi);        
        }
    }
#endif

    // SGD stuff
    // ---------
    int epoch = 0;
    double meanUGF = epsilon; // Not 0., otherwise in principle we could divide by zero later
    double meanUIR = epsilon;
    double meanUPR = epsilon;
    double U_GF0 = 1.;
    double U_IR0 = 1.;
    double U_PR0 = 1.;
    double LearningRate[maxUnits];
    double LearningRateC = StartLR; 
    for (int id=0; id<Nunits; id++) {
        LearningRate[id] = StartLR;
    }
    // Scale maximum displacement with size of array and not anymore by fixed fraction of DetectorSpacing
    // --------------------------------------------------------------------------------------------------
    static double EffectiveSpacing; 
    //double maxDispl = DisplFactor*DetectorSpacing;
    //if (shape==3) {
    //    maxDispl = DisplFactor*SpacingStep;  // max step in R during SGD
    //}
    double maxDispl2; // = pow(maxDispl,2); 


#ifdef PLOTS
    TCanvas * C1 = new TCanvas ("C1","",1000,500);
    C1->Divide(4,2);
    C1->cd(1);
    DXP->Draw();
    C1->cd(2);
    DYP->Draw();
    C1->cd(3);
    DXG->Draw();
    C1->cd(4);
    DYG->Draw();
    C1->cd(5);
    DTHP->Draw();
    C1->cd(6);
    DPHP->Draw();
    C1->cd(7);
    DTHG->Draw();
    C1->cd(8);
    DPHG->Draw();
    C1->cd(9);
    DTHPvsT->Draw("COLZ");
    C1->cd(10);
    DTHGvsT->Draw("COLZ");
    C1->Update();
#endif

    // Define canvas for temporary plots here
    // --------------------------------------
    TCanvas * CT = nullptr; // = new TCanvas ("CT","",1400,650);
    char namepng[120];
#ifdef FEWPLOTS
    // If scanU is on, this is a canvas for histos of residuals in X0, Y0
    // ------------------------------------------------------------------
    TCanvas * C0 = nullptr; // = new TCanvas ("C0","",500,500); 
    // C0->Update();
#endif

#ifdef STANDALONE
    sprintf (namepng,"/lustre/cmswork/dorigo/swgo/MT/Layouts/Layout_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, epoch+startEpoch);
#endif
#ifdef UBUNTU
    sprintf (namepng,"/home/tommaso/Work/swgo/MT/Layouts/Layout_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, epoch+startEpoch);
#endif
#ifdef INROOT
    sprintf (namepng,"./SWGO/Layouts/Layout_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, epoch+startEpoch);
#endif
    // CT->Print(namepng);

    // In case we are studying the utility as a function of the position of one detector unit,
    // (scanU = true), we specify here the relevant quantities. The 2d histogram filled is Uvsxy.
    // ------------------------------------------------------------------------------------------
    double x0 = x[idstar];
    double y0 = y[idstar];
    int ind_xincr;
    int ind_yincr;
    int side = (int)(sqrt(Nepochs));
    double sumUGF  = 0.; // Useful for noSGDupdate runs
    double sumUIR  = 0.;
    double sumUPR  = 0.;
    double sumUTC  = 0.;
    double sumUPS  = 0.;
    double sumUGF2 = 0.; // Useful for noSGDupdate runs
    double sumUIR2 = 0.;
    double sumUPR2 = 0.;
    double sumUTC2 = 0.;
    double sumUPS2 = 0.;
    double maxfx   = 100.;
    if (UseAreaCost) {
        cout    << " Initial area of layout = " << ConvexHull() << endl << endl;
        outfile << " Initial area of layout = " << ConvexHull() << endl << endl;
    }

    // If we need to run for a large number of epochs, as when we are scanning the utility in the 
    // xy space of detector idstar, or for long optimization runs, we encounter a problem due to
    // some memory leak of unidentified origin killing the job after 374 iterations. We bypass 
    // the problem by not using any plots within the SGD loop in that case.
    // ------------------------------------------------------------------------------------------ 
    bool noplots = false;
    //if (Nepochs>370) noplots = true;

    // Beginning of big optimization loop
    // ----------------------------------
    do { // SGD

        // Special runs for checks of gradients will compute these 
        // NB define x,y [idstar] _before_ you recompute TotalRspan,
        // lest the utility_GF is computed incorrectly by using the
        // previous Exposure!
        // -------------------------------------------------------
        if (scanU) {
            ind_xincr = epoch%side;
            ind_yincr = epoch/side;
            // Study neighborhood of a detector in terms of U values:
            // Modify the position of this detector to recompute U at different locations
            // --------------------------------------------------------------------------
            x[idstar] = x0 -rangex + 2.*rangex*(0.5+ind_xincr)/side; 
            y[idstar] = y0 -rangey + 2.*rangey*(0.5+ind_yincr)/side;
        }

        // Define span x and y of generated showers to illuminate layout
        // -------------------------------------------------------------
        Rspan(); // Fills ArrayRspan[0-2] values
        TotalRspan = ArrayRspan[0]+Rslack;
        // cout << "    R span is " << TotalRspan << endl;

        // Scale maximum displacement with size of array and not anymore by fixed fraction of DetectorSpacing
        // --------------------------------------------------------------------------------------------------
        EffectiveSpacing = sqrt(pi/Nunits)*(ArrayRspan[1]+ArrayRspan[2]); // We try mean+1sigma as effective radius for this purpose
        maxDispl = DisplFactor*EffectiveSpacing; 
        if (maxDispl>400.) maxDispl = 400.;
        //double maxDispl = DisplFactor*DetectorSpacing;
        //if (shape==3) {
        //    maxDispl = DisplFactor*SpacingStep;  // max step in R during SGD
        //}
        maxDispl2 = pow(maxDispl,2.); 

        // We define a correction to the flux-uncertainty part of the Utility as
        // a function of the effective density of showers generated on the ground
        // in a batch. This is N/area but N is Nactive, and the area needs to account
        // for the generation of showers within a radius Rtot=TotalRspan and the
        // accept/reject procedure depending on showers being closer than Rslack from
        // at least one detector. So Area = pi*Rtot^2 * Nbatch/Ntrials, and we come
        // up with rho = Nactive*Ntrials/(Rtot^2*Nbatch). The correction has to be
        // of the kind 1/sqrt(rho), as utility scales with square root of integrated
        // time - or shower density on the ground
        // ----------------------------------------------------------------------------
        ExposureFactor = TotalRspan; // The other factors are picked up later
        // note that if CircleExposure is false, the above def still applies, as the
        // illuminated area in the trials procedure still scales with the above.
    
        // Now we can redefine plots that depend on ArrayRspan
        // ---------------------------------------------------
        for (int id =0; id<Nunits; id++) {
            if (x[id]<-maxfx) maxfx = -x[id];
            if (x[id]>maxfx)  maxfx = x[id];
            if (y[id]<-maxfx) maxfx = -y[id];
            if (y[id]>maxfx)  maxfx = y[id];
        }

        //if (!noplots) {

            if (Layout != nullptr) { // Avoid memory leaks
                //gROOT->GetListOfHistograms()->Remove(Layout);
                //gROOT->GetListOfHistograms()->Remove(Layout2);
                //gROOT->GetListOfHistograms()->Remove(Showers3);
                //gROOT->GetListOfHistograms()->Remove(Showers3p);
                //gROOT->GetListOfHistograms()->Remove(Rdistr0);
                //gROOT->GetListOfHistograms()->Remove(Rdistr);
                //gROOT->GetListOfHistograms()->Remove(Pdistr0);
                //gROOT->GetListOfHistograms()->Remove(Pdistr);
                //gROOT->GetListOfHistograms()->Remove(NumStepsvsxy);
                //gROOT->GetListOfHistograms()->Remove(NumStepsvsxyN);
                delete Layout;
                delete Layout2;
                delete Showers3;
                //delete Pampalabola;
                delete Line1;
                delete Line2;
                delete Line3;
                if (!plotdensity) {
                    delete Showers3p;
                }
                delete Rdistr0;
                delete Rdistr;
                delete Pdistr0;
                delete Pdistr;
                delete NumStepsvsxy;
                delete NumStepsvsxyN;
                Layout        = nullptr;
                Layout2       = nullptr;
                Showers3      = nullptr;
                //Pampalabola   = nullptr;
                Showers3p     = nullptr;
                Line1         = nullptr;
                Line2         = nullptr;
                Line3         = nullptr;
                Rdistr0       = nullptr;
                Rdistr        = nullptr;
                Pdistr0       = nullptr;
                Pdistr        = nullptr;
                NumStepsvsxy  = nullptr;
                NumStepsvsxyN = nullptr;
            }
            Layout         = new TH2D       ("Layout",   "Layout of array", 500, -TotalRspan, TotalRspan, 500, -TotalRspan, TotalRspan);
            Layout2        = new TH2D       ("Layout",   "Layout of array", 500, -maxfx*1.05, maxfx*1.05, 500, -maxfx*1.05, maxfx*1.05);
            if (plotdensity) {
                Showers3   = new TH2D       ("Showers3", "Distribution of shower cores",  200, -TotalRspan, TotalRspan, 200, -TotalRspan, TotalRspan);
            } else {
                Showers3   = new TH2D       ("Showers3",  "Distribution of shower cores",  20, -TotalRspan, TotalRspan, 20, -TotalRspan, TotalRspan);
                Showers3p  = new TProfile2D ("Showers3p", "Measurement error of shower cores",  20, -TotalRspan, TotalRspan, 20, -TotalRspan, TotalRspan, 0., 1000.);
            }
            //Pampalabola    = TImage::Open("./SWGO/Detectors/pampalabola.jpg");
            Rdistr0        = new TH1D       ("Rdistr0",  "R distribution of detectors", NbinsRdistr, 0., TotalRspan);
            Rdistr         = new TH1D       ("Rdistr",   "R distribution of detectors", NbinsRdistr, 0., TotalRspan);
            Pdistr0        = new TH1D       ("Pdistr0",  "Phi distribution of detectors", NbinsPdistr, 0., twopi);
            Pdistr         = new TH1D       ("Pdistr",   "Phi distribution of detectors", NbinsPdistr, 0., twopi);
            NumStepsvsxy   = new TH2D       ("NumStepsvsxy","",20,0.,TotalRspan,20,0.1,10.);
            NumStepsvsxyN  = new TH2D       ("NumStepsvsxyN","",20,0.,TotalRspan,20,0.1,10.);
            Layout->SetMarkerStyle(20);
            Layout->SetMarkerColor(kRed);
            Layout->SetMarkerSize(0.4);
            Layout2->SetMarkerStyle(20);
            Layout2->SetMarkerColor(kRed);
            Layout2->SetMarkerSize(0.5);
#ifdef UBUNTU
            Layout->SetMarkerSize(2.);
            Layout2->SetMarkerSize(2.);
#endif
            Rdistr0->SetLineColor(kRed);
            Rdistr0->SetLineWidth(3);
            Rdistr->SetLineWidth(3);
            Rdistr->SetMinimum(0);
            Pdistr0->SetLineColor(kRed);
            Pdistr0->SetLineWidth(3);
            Pdistr->SetLineWidth(3);
            Pdistr->SetMinimum(0);

            // Layout and R distribution are already set
            // -----------------------------------------
            for (int id=0; id<Nunits; id++) {
                if (PlotThis[1]) Layout->Fill(x[id],y[id]);
                // Reference R distribution
                // ------------------------
                if (PlotThis[2]) {
                    double r = sqrt(xinit[id]*xinit[id]+yinit[id]*yinit[id]); // as radius expands, histo also changes
                    // if (r==0) r = 1.;
                    Rdistr0->Fill(r); // Fill(r,1./(twopi*r));
                }
                if (PlotThis[12]) Layout2->Fill(x[id],y[id]);
                //if (PlotThis[11]) Pdistr0->Fill(PhiFromXY(xinit[id],yinit[id])); // as radius expands, histo also changes
            }
        
        //} // End !noplots

        // Since we are varying the radius within which we generate showers as we go,
        // we reset the truex0, truey0 of the showers if they are not randomly generated
        // Note that since IsGamma[] indicates a photon for even is, and
        // a proton for odd is (see below when GenerateShower is called),
        // we are alternating photons and protons on the same radii. This
        // should be changed if other geometries are concerned, in case it
        // may interfere with correct placement of detector units.
        // ---------------------------------------------------------------
        if (fixShowerPos) SetShowersXY ();

#ifdef PLOTS
        // Reset histograms tracking goodness of position fits and others updated per epoch
        // --------------------------------------------------------------------------------
        DXG->Reset();
        DYG->Reset();
        DXP->Reset();
        DYP->Reset();
        PG->Reset();
        PP->Reset();
#endif
        // Plots that get refilled at every epoch need a reset here
        // --------------------------------------------------------
        //if (PlotThis[13]) ThGIvsThGP->Reset();
        //if (PlotThis[14]) LrGIvsLrGP->Reset();
        //if (!noplots) {
            if (!usetrueE) {
                if (PlotThis[5]) DE->Reset();
                if (PlotThis[9]) {
                    HEtrue->Reset();
                    HEmeas->Reset();
                }
            }
            if (!usetrueAngs) {
                if (PlotThis[2]) DR->Reset();
            }
        //} // End if !noplots
        outfile << endl;
        cout    << endl;
        if (Nthreads==1) {
            outfile << "     Event reconstruction # ";
            cout    << "     Event reconstruction # ";
        }
        Ng_active     = 0;
        Np_active     = 0;
        AverLastXIncr = 0.; // Check convergence of lnL E maximization
        AverLastYIncr = 0.; // Check convergence of lnL E maximization
        AverLastEIncr = 0.; // Check convergence of lnL E maximization
        AverLastTIncr = 0.; // Check convergence of lnL E maximization
        AverLastPIncr = 0.; // Check convergence of lnL E maximization
        NAverLastIncr = 0;

#if defined(STANDALONE) || defined(UBUNTU)

        // If we generate the same showers, we need to run GenerateShower first for Nevents, then for Nbatch
        // ones, as the latter will require that TrueX0[], TrueY0[] be defined already.
        // -------------------------------------------------------------------------------------------------
        if (SameShowers) {
            for (int i=0; i<Nthreads; ++i) {
                threads.emplace_back (threadFunction, i);
            }
            // Wait for all threads to finish
            // ------------------------------
            for (auto& thread : threads) {
                if (thread.joinable()) thread.join();
            }
            threads.clear();
            for (int i=Nthreads; i<2*Nthreads; ++i) {
                threads.emplace_back (threadFunction, i);
            }             
            // Wait for all threads to finish
            // ------------------------------
            for (auto& thread : threads) {
                if (thread.joinable()) thread.join();
            }
        } else {
            for (int i=0; i<Nthreads; ++i) {
                threads.emplace_back (threadFunction, i);
            }
            // Wait for all threads to finish
            // ------------------------------
            for (auto& thread : threads) {
                if (thread.joinable()) thread.join();
            }
            threads.clear(); 
        }

#endif
#ifdef INROOT
        threadFunction (0); // there is only one thread, #0
#endif

        // Statistics reporting on shower likelihood convergence
        // -----------------------------------------------------
        cout << "     Aver. last X incr. = " << (AverLastXIncr+AverLastYIncr)/(2.*NAverLastIncr);
        cout << " E incr. = " << AverLastEIncr/NAverLastIncr;
        cout << " T incr. = " << AverLastTIncr/NAverLastIncr;
        cout << " P incr. = " << AverLastPIncr/NAverLastIncr; 
        cout << " N steps = " << NumAvgSteps/DenAvgSteps << endl;
        //outfile << "     Aver. last X incr. = " << (AverLastXIncr+AverLastYIncr)/(2.*NAverLastIncr);
        //outfile << " E incr. = " << AverLastEIncr/NAverLastIncr;
        //outfile << " T incr. = " << AverLastTIncr/NAverLastIncr;
        //outfile << " P incr. = " << AverLastPIncr/NAverLastIncr; 
        //outfile << " N steps = " << NumAvgSteps/DenAvgSteps << endl;

        // Now collect some info on the showers after possible multithreading on is
        // ------------------------------------------------------------------------
        for (int ibin=0; ibin<20; ibin++) {
            meandE_E[ibin]  = 0.;
            meandE2_E[ibin] = 0.;
            denE_E[ibin]    = 0.;
            meandR_E[ibin]  = 0.;
            meandR2_E[ibin] = 0.;
            denR_E[ibin]    = 0.;
            meandE_R[ibin]  = 0.;
            meandE2_R[ibin] = 0.;
            denE_R[ibin]    = 0.;
            meandR_R[ibin]  = 0.;
            meandR2_R[ibin] = 0.;
            denR_R[ibin]    = 0.;
            for (int jbin=0; jbin<20; jbin++) {
                meandE[ibin][jbin] = 0.;
                meandE2[ibin][jbin]= 0.;
                denE[ibin][jbin]   = 0.;
                meandR[ibin][jbin] = 0.;
                meandR2[ibin][jbin]= 0.;
                denR[ibin][jbin]   = 0.;
            }
        }
        for (int is=0; is<Nevents+Nbatch; is++) {
            if (!Active[is]) continue;

            if (is>=Nevents) {
                if (IsGamma[is]) {
                    Ng_active += PActive[is];
                } else {
                    Np_active += PActive[is];
                }
            }
            // Also fill shower 2d distribution with error on position
            // -------------------------------------------------------
            //int Ebin = (int)(TrueE[is]*2.);
            // We fill the histograms of energy res. as a function of log of energy
            // in 20 bins from -1 to +1. -1 corresponds to log(0.1PeV) and +1 to log(10 PeV).
            // To map this we do the following:
            // since log(0.1)=-log_10 and log(10)=log_10, we divide by log_10:
            // Ebin = 20*log(E)/log_10
            // To get the energy from a bin, we later will need to invert this:
            // Ebin/20*log_10 = log(E) -> E = exp((Ebin+0.5)*log_10/20) 
            // --------------------------------------------------------
            if (IsGamma[is]) {

#ifdef PLOTRESOLUTIONS
                // Get MSE of energy and angle
                // ---------------------------
                int Ebin     = (int)(2.*TrueE[is]); // From 0. to 10. -> from 0 to 20
                double Rcore = sqrt(pow(TrueX0[is],2.)+pow(TrueY0[is],2.));
                int Rbin     = (int)(5.*(log(Rcore)-4.)); // From 4 to 8 -> from 0 to 20
                double dp    = pi-fabs(fabs(TruePhi[is]-phmeas[is][0])-pi);
                double dR    = sqrt(pow(TrueTheta[is]-thmeas[is][0],2.)+pow(sin(TrueTheta[is])*dp,2.));
                double dE    = (e_meas[is][0]-TrueE[is])/TrueE[is];
                if (Ebin>=0 && Ebin<20 && Rbin>=0 && Rbin<20) {
                    meandE2[Ebin][Rbin] += PActive[is]*dE*dE;
                    meandE[Ebin][Rbin]  += PActive[is]*dE;
                    denE[Ebin][Rbin]    += PActive[is];
                    meandR2[Ebin][Rbin] += PActive[is]*dR*dR;
                    meandR[Ebin][Rbin]  += PActive[is]*dR;
                    denR[Ebin][Rbin]    += PActive[is];
                }
                if (Ebin>=0 && Ebin<20) {
                    meandE2_E[Ebin] += PActive[is]*dE*dE;
                    meandE_E[Ebin]  += PActive[is]*dE;
                    denE_E[Ebin]    += PActive[is];
                    meandR2_E[Ebin] += PActive[is]*dR*dR;
                    meandR_E[Ebin]  += PActive[is]*dR;
                    denR_E[Ebin]    += PActive[is];
                }
                if (Rbin>=0 && Rbin<20) {
                    meandE2_R[Rbin] += PActive[is]*dE*dE;
                    meandE_R[Rbin]  += PActive[is]*dE;
                    denE_R[Rbin]    += PActive[is];
                    meandR2_R[Rbin] += PActive[is]*dR*dR;
                    meandR_R[Rbin]  += PActive[is]*dR;
                    denR_R[Rbin]    += PActive[is];
                }
#endif
                //if (!noplots) {
                    double Derror = sqrt(pow(TrueX0[is]-x0meas[is][0],2)+pow(TrueY0[is]-y0meas[is][0],2));
                    double Eerror = fabs(e_meas[is][0]-TrueE[is])/TrueE[is]; // 1./(TrueE[is]*InvRmsE[is]); // fabs(e_meas[is][0]-TrueE[is])/TrueE[is];
                    if (Eerror>1.) Eerror = 1.; // Cap the deviation
                    if (PlotThis[1]) {
                        if (plotdensity) {
                            Showers3->Fill(TrueX0[is],TrueY0[is]);
                        } else if (!usetrueXY) {
                            Showers3p->Fill(TrueX0[is],TrueY0[is],Derror);
                        } else {
                            Showers3p->Fill(TrueX0[is],TrueY0[is],Eerror);
                        }
                    }
                    if (!usetrueE) {
                        if (PlotThis[5]) {
                            if (epoch==0) DE0->Fill(TrueE[is],Eerror,PActive[is]);
                            DE->Fill(TrueE[is],Eerror,PActive[is]);
                        }
                        if (PlotThis[9]) {
                            HEtrue->Fill(TrueE[is]);
                            HEmeas->Fill(e_meas[is][0]);
                        }
                    }
                    if (!usetrueAngs && PlotThis[4]) {
                        double dp = pi-fabs(fabs(TruePhi[is]-phmeas[is][0])-pi);
                        double dR = sqrt(pow(TrueTheta[is]-thmeas[is][0],2)+pow(sin(TrueTheta[is])*dp,2)+epsilon);
                        if (epoch==0) {
                            if (PlotThis[4]) DR0->Fill(TrueE[is],log(dR));
                        } 
                        DR->Fill(TrueE[is],log(dR));
                    }
                //} // End if !noplots
            }
        } // End is loop on all generated events

        // Operation done only at loop 0, or if we are looping to get more data
        // --------------------------------------------------------------------
        if (epoch==0 || noSGDupdate) {
#ifdef PLOTRESOLUTIONS            
            for (int ibin=0; ibin<20; ibin++) {
                double eb = 0.25+0.5*ibin; // We divided energies in 20 bins above
                // Deal with 1D profiles vs E
                if (denE_E[ibin]>0.) {
                    meandE2_E[ibin] = (meandE2_E[ibin]/denE_E[ibin] - pow(meandE_E[ibin]/denE_E[ibin],2.));
                    DE0vsE->Fill(eb,sqrt(meandE2_E[ibin]));
                }
                if (denR_E[ibin]>0.) {
                    meandR2_E[ibin] = (meandR2_E[ibin]/denR_E[ibin] - pow(meandR_E[ibin]/denR_E[ibin],2.));
                    DR0vsE->Fill(eb,sqrt(meandR2_E[ibin]));
                }
                double rb = 0.1+4.+0.2*ibin; // 4-8 in 20 bins
                if (denE_R[ibin]>0.) {
                    meandE2_R[ibin] = (meandE2_R[ibin]/denE_R[ibin] - pow(meandE_R[ibin]/denE_R[ibin],2.));
                    DE0vsR->Fill(rb,sqrt(meandE2_R[ibin]));
                }
                if (denR_R[ibin]>0.) {
                    meandR2_R[ibin] = (meandR2_R[ibin]/denR_R[ibin] - pow(meandR_R[ibin]/denR_R[ibin],2.));
                    DR0vsR->Fill(rb,sqrt(meandR2_R[ibin]));
                }
                   
                for (int jbin=0; jbin<20; jbin++) {
                    rb = 0.1+4.+0.2*jbin; // 4-8 in 20 bins. 
                    if (denE[ibin][jbin]>0.) {
                        meandE2[ibin][jbin] = (meandE2[ibin][jbin]/denE[ibin][jbin] - pow(meandE[ibin][jbin]/denE[ibin][jbin],2.));
                    }
                    if (denR[ibin][jbin]>0.) {
                        meandR2[ibin][jbin] = (meandR2[ibin][jbin]/denR[ibin][jbin] - pow(meandR[ibin][jbin]/denR[ibin][jbin],2.));
                    }
                    if (meandE2[ibin][jbin]>0.)  
                        DE0vsER->Fill(rb,eb,sqrt(meandE2[ibin][jbin]));
                        NvsER->Fill(rb,eb,denE[ibin][jbin]);
                    if (meandR2[ibin][jbin]>0.) 
                        DR0vsER->Fill(rb,eb,sqrt(meandR2[ibin][jbin]));
                }
            }
#endif
        }
        // Now events have been declared inactive if they have too bad reconstruction, so we compute the true g fraction
        // -------------------------------------------------------------------------------------------------------------
        if (Ng_active==0.) {
            outfile << "     Sorry, no photon accepted. Terminating." << endl;
            cout    << "     Sorry, no photon accepted. Terminating." << endl;
            warnings3++;
            TerminateAbnormally();
            return 0;
        }
        N_active = Ng_active+Np_active;
        TrueGammaFraction = Ng_active/N_active; // For batch events only

        // Compute stuff that does not depend on detector positions
        // E.g. inverse variance, which is minus the second derivative
        // of the log likelihood versus signal fraction. As the log L is written
        //     log L = Sum_i { MeasFg * Pg + (1-MeasFg)*Pp}, 
        // we get
        //     d^2 logL / dMeasFg^2 = - Sum_i {(Pg-Pp)/(MeasFg*Pg+(1-MeasFg)*Pp)}^2 
        // whence
        //     sigma^2 = 1/Sum_i{...}^2 
        // So we construct the probability density functions Pg[k], Pp[k] for
        // each batch event k, by looping on 1:Nevents and adding Gaussian kernels.
        // ------------------------------------------------------------------------

        // Compute the PDF of the test statistic for all batch showers
        // -----------------------------------------------------------
        // double JS = 0.;
        int Nsmallp = 0;
        for (int k=Nevents; k<Nevents+Nbatch; k++) {
            if (!Active[k]) continue;
            pg[k] = ComputePDF (k,true);
            pp[k] = ComputePDF (k,false);
            double m_x = 0.5*(pg[k]+pp[k]);
            // if (Pg>0. && Pp>0.) JS += 0.5 * (Pg*log(Pg/m_x)+Pp*log(Pp/m_x));
            if (pg[k]<epsilon && pp[k]<epsilon) Nsmallp++;
#ifdef PLOTS
            // Histograms of PDFs:
            // -------------------
            double pgstar = pg[k];
            if (pgstar<epsilon) pgstar = epsilon;
            double ppstar = pp[k];
            if (ppstar<epsilon) ppstar = epsilon;
            PG->Fill(log(pgstar));
            PP->Fill(log(ppstar));
#endif
        }
        if (1.*Nsmallp/N_active>0.01) {
            cout    << "Warning, " << 100.*Nsmallp/N_active << "% of active showers have negligible estimated pdf. " << endl;
            outfile << "Warning, " << 100.*Nsmallp/N_active << "% of active showers have negligible estimated pdf. " << endl;
            warnings6++;
        }
        // Compute the gamma fraction in this batch by zeroing the lnL derivative
        // ----------------------------------------------------------------------
        MeasFg = MeasuredGammaFraction(); // Also computes static inv_sigmafs2

        if (inv_sigmafs2==0.) {
            inv_sigmafs2 = epsilon;
            cout    << "Warning, inv_sigmafs2 = 0" << endl;
            outfile << "Warning, inf_sigmafs2 = 0" << endl;
            warnings1++;
            TerminateAbnormally();
            return 0;            
        }
        sigmafs2 = 1./inv_sigmafs2;
        MeasFgErr = sqrt(sigmafs2);
        inv_sigmafs  = 1./MeasFgErr; // Above we were computing the variance with RCF bound
        outfile << "     Ng, Np active in this batch = " << Ng_active << " " << Np_active << endl; 
        cout    << "     Ng, Np active in this batch = " << Ng_active << " " << Np_active << endl; 
        if (Ng_active<10.) {
            cout    << "      Sorry, too few reconstructable showers. terminating. " << endl;
            outfile << "      Sorry, too few reconstructable showers. terminating. " << endl;
            warnings3++;
            TerminateAbnormally();
            return 0;
        }

        // Construct template of LLRT for gamma and proton, for first ie bin
        // -----------------------------------------------------------------
        double minLRT = -1000.; // Was -100000
        /*
        double minLRT = largenumber;
        double maxLRT = -largenumber;
        for (int is=0; is<Nevents; is++) {
            double sqm = sigmaLRT[0][is];
            if (logLRT[0][is]-2.*sqm<minLRT) minLRT = logLRT[0][is]-2.*sqm;
            if (logLRT[0][is]+2.*sqm>maxLRT) maxLRT = logLRT[0][is]+2.*sqm;
        }
        */
#ifdef FEWPLOTS
        //if (!noplots) {
            if (LLRP!=nullptr) {
                //gROOT->GetListofHistograms->Remove(LLRP);
                //gROOT->GetListofHistograms->Remove(LLRG);
                delete LLRP;
                delete LLRG;
                LLRP = nullptr;
                LLRG = nullptr;
            }
            LLRP = new TH1D ("LLRP", "Log-likelihood ratio", 350, 3., 10.); // log(maxLRT-minLRT+1.)+1); 
            LLRG = new TH1D ("LLRG", "Log-likelihood ratio", 350, 3., 10.); // log(maxLRT-minLRT+1.)+1);
            LLRG->SetLineWidth(1);
            LLRP->SetLineWidth(1);
            LLRP->SetLineColor(kRed);
            if (PlotThis[6]) {
                for (int is=0; is<Nevents; is++) {
                    if (!Active[is]) continue;
                    // Smooth LLRT distributions with Gaussian kernel
                    // by sampling 100 times a Gaussian for every logLRT value
                    // -------------------------------------------------------
                    for (int irnd=0; irnd<1000000/Nevents; irnd++) {
                        double thisLRT = logLRT[is]+shift[irnd]*sigmaLRT[is];
                        if (IsGamma[is]) { // Gamma event
                            if (thisLRT>=minLRT) {
                                LLRG->Fill(log(thisLRT-minLRT+1.));
                            }
                        } else { // Proton event
                            if (thisLRT>=minLRT) {
                                LLRP->Fill(log(thisLRT-minLRT+1.));
                            }
                        }
                    }
                }
            }
        //} // End if !noplots
#endif
        // Now we can set the renormalizing factor for the utility
        // -------------------------------------------------------
        if (CircleExposure && !scanU) { // Otherwise Ntrials is not filled
            totNtrials = 0;
            for (int is=Nevents; is<Nevents+Nbatch; is++) {
                totNtrials += Ntrials[is];
            }
            // This below is the squared root of the inverse density, which we use to 
            // correct the utility for the exposure time. We do U = [flux/sigma(flux)]/sqrt(rho)
            // as the stat uncertainty on the utility should scale with the inverse square root
            // of the integration time == density of gen shower in a batch.
            // We have defined ExposureFactor = Rtot, now we multiply it by sqrt(successes/trials)
            // to correct the radius (Area^0.5) by the effective coverage of detectors. Since successes
            // are represented by Nbatch (we do not care if fewer or more showers get to trigger, because
            // this is included in the flux uncertainty at the end, and it has nothing to do with the time
            // integration that Ntrials / Nbatch corresponds to), we can boil it down to Rtotal/sqrt(Ntrials).
            // --------------------------------------------------------------------------------------------------
            if (totNtrials>0.) {
                ExposureFactor *= sqrt(1./totNtrials); // sqrt(1.*Nbatch/totNtrials/N_active);
            } else {
                warnings1++;
                cout    << "    Zero totNtrials ?" << endl;
                outfile << "    Zero totNtrials ?" << endl;
                TerminateAbnormally();
                return 0;
            }
        } else {
            // Otherwise we correct ExposureFactor by the number of events
            // -----------------------------------------------------------
            ExposureFactor *= sqrt(1./Nbatch); 
        }

        // Below we compute utility function 
        // ---------------------------------
        cout    << "     GF = " << MeasFg << " +- " << MeasFgErr << " (true = " << TrueGammaFraction << ")"; 
        outfile << "     GF = " << MeasFg << " +- " << MeasFgErr << " (true = " << TrueGammaFraction << ")"; 
        if (!usetrueXY)   ComputeUtilityGF();
        if (!usetrueE)    ComputeUtilityIR();
        if (!usetrueAngs) ComputeUtilityPR();
        if (PeVSource) { 
            ComputeUtilityPeVSource(); // The relative utility is technically still related to flux, we keep it under U_GF
        } 
        double U_TC = 0.;
        if (UseAreaCost) {
            ComputeUtilityArea(); // Computes U_TA if default mode=0
            U_TC += U_TA;
        }
        if (UseLengthCost) {
            ComputeUtilityLength();
            U_TC += U_TL;
        }
        Utility = 0.;
        if (PeVSource) {
            Utility = U_PS; // + U_TC;  // Leave cost out of total utility, only use it as a derivative contribution
        } else {
            if (eta_GF>0.) Utility += U_GF;
            if (eta_IR>0.) Utility += U_IR;
            if (eta_PR>0.) Utility += U_PR;
            // if (eta_TL + eta_TA>0.) Utility += U_TC; // Leave cost out of total utility, only use it as a derivative contribution
        }
        sumUGF  += U_GF;
        sumUIR  += U_IR;
        sumUPR  += U_PR;
        sumUTC  += U_TC;
        sumUPS  += U_PS;
        sumUGF2 += U_GF*U_GF;
        sumUIR2 += U_IR*U_IR;
        sumUPR2 += U_PR*U_PR;
        sumUTC2 += U_TC*U_TC;
        sumUPS2 += U_PS*U_PS;
        cout    << endl;  // cout    << " JS div. = " << JS << endl;
        outfile << endl;  // outfile << " JS div. = " << JS << endl;

        // Fill Pareto front histogram
        // ---------------------------
        Pareto->Fill(U_IR,U_PR);

#ifdef PLOTS
        LLRP = new TH1D ("LLRP", "Log-likelihood ratio", 500, 0., 12.); // log(maxLRT-minLRT+1.)+1); 
        LLRG = new TH1D ("LLRG", "Log-likelihood ratio", 500, 0., 12.); // log(maxLRT-minLRT+1.)+1);
        LLRG->SetLineWidth(1);
        LLRP->SetLineWidth(1);
        LLRP->SetLineColor(kRed);
        U_gf->SetBinContent (epoch+1, U_GF);
        U_ir->SetBinContent (epoch+1, U_IR);
        U_pr->SetBinContent (epoch+1, U_PR);
        U_tc->SetBinContent (epoch+1, U_TC);
#endif
#ifdef FEWPLOTS
        if (scanU) { 
            if (C0!=nullptr) {
                delete C0;
            }
            C0 = new TCanvas ("C0","",1600,500);
            Uvsxy->Fill(x[idstar],y[idstar],Utility);
            if (fabs(y[idstar]-y0)<rangey/side) Uvsx->Fill(x[idstar],Utility);
            if (fabs(x[idstar]-x0)<rangex/side) Uvsy->Fill(y[idstar],Utility);
            C0->Divide(3,1);
            C0->cd(1);
            Uvsxy->Draw("COLZ");
            C0->cd(2);
            Uvsx->Draw();
            C0->cd(3);
            Uvsy->Draw();
            C0->Update();
            
            epoch++;
            continue;
        }
        // if (Utility<0.) Utility = 0.;
        // if (Utility>MaxUtility && epoch>0) Utility = U->GetBinContent(epoch); // use previous value to avoid messing up the U graph
#endif
        //if (!noplots) {
            if (PlotThis[0]) {
                U->SetBinContent(epoch+1,Utility);
                Uave->Fill(epoch+0.5,Utility);
                if (Utility>maxUtility) {
                    maxUtility = Utility;
                    imax = epoch;
                }
                U->SetMaximum(1.1*maxUtility);
            }
            // We fill Utility ratio plots only after we are done with the first bin, set at 1.
            if (PlotThis[3]) {
                if (epoch<NepPerBin) {
                    meanUGF += U_GF;
                    meanUIR += U_IR;
                    meanUPR += U_PR;
                    if (epoch==NepPerBin-1) {
                        U_GF0 = meanUGF/NepPerBin;
                        U_IR0 = meanUIR/NepPerBin;
                        U_PR0 = meanUPR/NepPerBin;
                    }
                } else {
                    UaveGF->Fill(epoch,U_GF/U_GF0);
                    UaveIR->Fill(epoch,U_IR/U_IR0);
                    UavePR->Fill(epoch,U_PR/U_PR0);
                    UaveTC->Fill(epoch,(U_GF+U_IR+U_PR)/(U_GF+U_IR+U_PR+U_TC));
                }
            }
        //} // End if !noplots

        // Zero a few arrays
        // -----------------
        double aveDispl = 0.; // Keep track of average displacement at each epoch
        double displ[maxRbins];
        int Ndispl[maxRbins];
        double prev_displ[maxRbins];
        for (int ir=0; ir<NRbins; ir++) {
            displ[ir]      = 0.;
            prev_displ[ir] = 0.;
            Ndispl[ir]     = 0;
        }
        double commondx = 0.;
        double commondy = 0.;
        double avedUdR  = 0.;
        double Momentum_coeff = 0.05; // To be optimized
        double CosThetaEff    = 1.; // Effective angle for successive displacements, used to update learning rate

        // Comment from here if you want to bypass the SGD: /*
        // ---------------------------------------------------
        if (!noSGDupdate) {

            // Loop on detector units, to update detector positions following gradient of utility
            // ----------------------------------------------------------------------------------
            if (Nthreads==1) cout << "     Loop on detector # ";

            // Now get derivatives of U vs dx, dy for each detector, by splitting the job in threads.
            // --------------------------------------------------------------------------------------
#if defined(STANDALONE) || defined(UBUNTU)

            for (int i=0; i<Nthreads; ++i) {
                threads.emplace_back(threadFunction2, i);
            }

            // Wait for all threads to finish
            // ------------------------------
            for (auto& thread : threads) {
                if (thread.joinable()) thread.join();
            }
            threads.clear(); 
#endif
            // These can only be computed if not multithreading, else clashes in parallel exec may occur
            // -----------------------------------------------------------------------------------------
            // sumduc = 0;
            // sumdup = 0;

#ifdef INROOT
            threadFunction2 (0); // there is only one thread, #0
#endif
            if (Nthreads==1) {
                cout << endl;
                outfile << endl;
                // cout    << "     Average du ps = " << sumdup/Nunits << " du tc = " << sumduc/Nunits << endl; 
                // outfile << "     Average du ps = " << sumdup/Nunits << " du tc = " << sumduc/Nunits << endl;
            }
            // Compute average du/dR to renormalize size of steps
            // --------------------------------------------------
            for (int id=0; id<Nunits; id++) {
                avedUdR += sqrt(pow(dU_dxi[id],2.)+pow(dU_dyi[id],2.));
            }
            avedUdR /= Nactiveunits; 
            // For now we use Nactiveunits==Nunits, but if we want to exploit that functionality 
            // we need to check that keepfixed[id] is false for all units on which we operate

            //if (!noplots) {
                if (PlotThis[15]) {
                    // Compute average dUdR to see if detector is expanding or shrinking
                    // -----------------------------------------------------------------
                    double AvedU_dR = 0.;
                    for (int id=0; id<Nunits; id++) {
                        double xi = x[id];
                        double yi = y[id];
                        double Ri; 
                        double d2 = pow(xi,2.)+pow(yi,2.);
                        if (d2>0.) {
                            Ri = sqrt(d2);
                            double costheta = xi/(Ri+epsilon);
                            double sintheta = yi/(Ri+epsilon);
                            AvedU_dR += dU_dxi[id]*costheta + dU_dyi[id]*sintheta;
                        } else { // If detector is in (0,0), all the movement is a dR
                            AvedU_dR += sqrt(pow(dU_dxi[id],2.)+pow(dU_dyi[id],2.));
                        }
                    }
                    AvedU_dR /= Nactiveunits; // see note above for Nactiveunits and keepfixed[]
                    dUdR->Fill ((double)epoch,AvedU_dR);
                    // cout << "     avedudr = " << AvedU_dR << endl;
                }
            //} // end if !noplots            

            // Compute average of logs of components of du/dx for dynamic rescaling
            // Note, we needed arrays instead of sums on the fly, because of multithreading
            // ----------------------------------------------------------------------------
            if (DynamicLR) {
                double avelogdUgf = 0.;
                double avelogdUir = 0.;
                double avelogdUpr = 0.;
                double avelogdUtc = 0.;
                double ratioir = 1.;
                double ratiopr = 1.;
                double ratiotc = 1.;
                for (int id=0; id<Nunits; id++) {
                    avelogdUgf += ave_dUgf[id];
                    avelogdUir += ave_dUir[id];
                    avelogdUpr += ave_dUpr[id];
                    avelogdUtc += ave_dUtc[id];
                }
                avelogdUgf = avelogdUgf/Nunits; 
                avelogdUir = avelogdUir/Nunits;
                avelogdUpr = avelogdUpr/Nunits;
                avelogdUtc = avelogdUtc/Nunits;

                // Rescale etas. This also tries to equalize based on the utility value (experimental, v119)
                // -----------------------------------------------------------------------------------------
                if (eta_GF>0.) { 
                    if (eta_IR>0.) {
                        ratioir = exp(avelogdUgf)/exp(avelogdUir);
                        if (U_GF>0.) ratioir = ratioir * U_IR/U_GF;
                    }
                    if (eta_PR>0.) {
                        ratiopr = exp(avelogdUgf)/exp(avelogdUpr);
                        if (U_GF>0.) ratiopr = ratiopr * U_PR/U_GF;
                    }
                    if (avelogdUtc>0 && (eta_TA>0. || eta_TL>0.)) {
                        ratiotc = exp(avelogdUgf)/exp(avelogdUtc);
                        if (U_GF>0.) ratiotc = ratiotc * U_TC/U_GF;
                    }
                } else { // If no flux utility, rescale the other two one by the other
                    if (eta_PR>0.) {
                        ratiopr = exp(avelogdUir/avelogdUpr);
                        if (U_IR>0.) ratiopr = ratiopr * U_PR/U_IR;
                    }
                    if (avelogdUtc>0 && (eta_TA>0. || eta_TL>0.)) {
                        ratiotc = exp(avelogdUir/avelogdUtc);
                        if (U_IR>0.) ratiotc = ratiotc * U_TA/U_IR;
                    }
                }
                eta_IR = eta_IR * ratioir;
                eta_PR = eta_PR * ratiopr; 
                eta_TA = eta_TA * ratiotc;
                eta_TL = eta_TL * ratiotc;
                cout << "     EtaIR, etaPR, etaTC rescaled resp. by " << ratioir << ", " << ratiopr << ", and " << ratiotc << endl; 
            }
            
            // We could not update the positions as we went, because we were accumulating global increments. We do it now.
            // -----------------------------------------------------------------------------------------------------------
            double dx = 0.;
            double dy = 0.;
            double dr2, dr2prev;
            double f = LR_Scheduler (epoch);
            double k = 1.;
            if (avedUdR>0.) k = 0.25*maxDispl/avedUdR;
            aveDispl = 0.;

            // MixedMode is activated once at least one element is not in a multiplet anymore, because of action of ForbiddenRegion
            // --------------------------------------------------------------------------------------------------------------------
            if (!MixedMode && CommonMode==2) { // For now the mixed mode can only be activated if we start with CM=2
                for (int id=0; id<Nunits; id++) {
                    if (!InMultiplet[id]) MixedMode = true;
                }
            }
            if (CommonMode==0 || MixedMode) { // Do not vary R; vary independently x and y

                // Compute average displacement before equalization. We do it as follows:
                // 1) compute average gradient in x and y
                // 2) rescale movements to be 0.25 times the max allowed displacement, times a slowly decreasing scheduler
                // 3) multiply by an individual learning rate 
                // -------------------------------------------------------------------------------------------------------
                for (int id=0; id<Nunits; id++) {
                    if (InMultiplet[id]) continue; // This detector is part of a multiplet and is dealt with in CommonMode>1 part
                    if (scanU && id!=idstar) continue;
                    if (keep_fixed[id]) continue;

                    // Update independently each detector position based on gradient of U, ignoring 
                    // the symmetry of the problem
                    // ----------------------------------------------------------------------------
                    dx = k * f * dU_dxi[id] * LearningRate[id];
                    dy = k * f * dU_dyi[id] * LearningRate[id];

                    // Cap individual movements to be between 31% and 100% of maxdispl
                    // ---------------------------------------------------------------
                    float Current_mindispl2 = 0.1*f*f*maxDispl2; // We need to rescale the boundaries as the LR changes
                    float Current_maxdispl2 = 1.0*f*f*maxDispl2; 
                    dr2 = dx*dx+dy*dy;
                    if (dr2>Current_maxdispl2) {
                        dx  = dx*sqrt(Current_maxdispl2/dr2);
                        dy  = dy*sqrt(Current_maxdispl2/dr2);
                        dr2 = Current_maxdispl2;
                    } else if (dr2<Current_mindispl2) {
                        dx  = dx*sqrt(Current_mindispl2/dr2);
                        dy  = dy*sqrt(Current_mindispl2/dr2);
                        dr2 = Current_mindispl2;
                    }
                    if (PlotThis[10] || PlotThis[11]) {
                        dr2prev = pow(x[id]-xprev[id],2.) + pow(y[id]-yprev[id],2.);
                    }
                    
                    // Accumulate information on how consistent are the movements of detectors
                    // -----------------------------------------------------------------------
                    if (epoch>0) CosThetaEff = ((x[id]-xprev[id])*dx + (y[id]-yprev[id])*dy) /
                                               (sqrt(pow(x[id]-xprev[id],2)+pow(y[id]-yprev[id],2)+epsilon)*sqrt(pow(dx,2)+pow(dy,2)+epsilon));

                    // Ok, now update the positions
                    // ----------------------------
                    xprev[id] = x[id];
                    yprev[id] = y[id];
                    x[id] += dx;
                    y[id] += dy;
                    //cout << id << " " << dx << " " << dy << " " << dU_dxi[id] << " " << dU_dyi[id] << " " << LearningRate[id] << " " << CosThetaEff << endl;                    // Update learning rate based on "costhetaeff" value (see above) - this is the cosine

                    // Change LR based on the value                    
                    // of the angle between the current and the previous detector displacement. If positive,
                    // we increase the LR for that unit; if negative, we decrease it.
                    // -------------------------------------------------------------------------------------
                    double rate_modifier = -1.+2.*pow(0.5*(CosThetaEff+1.),2); 
                                           // the above function is -1 for x=-1, +1 for x=1, and -0.3 for x=0
                    LearningRate[id] *= exp (Momentum_coeff*rate_modifier);
                    // Clamp them - we do not want too much variation
                    // ----------------------------------------------
                    if (LearningRate[id]<MinLearningRate) LearningRate[id] = MinLearningRate;
                    if (LearningRate[id]>MaxLearningRate) LearningRate[id] = MaxLearningRate;

                    //if (!noplots) {
                    if (PlotThis[10] || PlotThis[11]) {
                        double ps = sqrt(dr2*dr2prev)/maxDispl2*CosThetaEff; // Renormalized scalar product
                        float sgnps = 1.;
                        if (ps<0.) sgnps = -1.;
                        ps = sgnps*pow(fabs(ps),0.25); // Blow up region around zero to enhance that area in plot
                        if (PlotThis[10]) {
                            if (epoch>0) {
                                double ac = acos(CosThetaEff);
                                CosDir->Fill(ps); // CosDir->Fill(ac);
                            }   
                        }
                        if (PlotThis[11]) {
                            if (epoch>0) {
                                double ac = acos(CosThetaEff);
                                CosvsEp->Fill(epoch,ac);
                            }
                        }
                    }
                    //} // End !noplots
#ifdef PLOTS
                    LR->Fill(epoch,log(LearningRate[id]));
#endif
                    if (dr2>0.) aveDispl += sqrt(dr2);
                    // cout << "     ID = " << id << ": x+dx,y+dy = " << x[id];
                    // if (dx>=0.) cout << "+";
                    // cout  << dx << "," << y[id];
                    // if (dy>=0.) cout << "+";
                    // cout << dy << endl;
                    if (scanU) {
                        cout << "xprev, x, dudx = " << xprev[idstar] << " " << x[idstar] << " " << dU_dxi[idstar] 
                            << " yprev, y, dudy = " << yprev[idstar] << " " << y[idstar] << " " << dU_dyi[idstar] << endl;
                    } 
                } // End id loop
            } 
            if (CommonMode==1) { // Vary R of detectors

                // Determine min and max distance for binning in r
                // -----------------------------------------------
                double rmin = largenumber;
                double rmax = 0.;
                for (int id=0; id<Nunits; id++) {
                    double r2 = pow(x[id],2.)+pow(y[id],2.);
                    if (r2>rmax) rmax = r2;
                    if (r2<rmin) rmin = r2;
                }
                if (rmax>0.) rmax = sqrt(rmax)+1.;
                if (rmin>0.) rmin = sqrt(rmin)-1.;
                if (rmin<0.) rmin = 0.;
                double rspan = rmax-rmin;

                // Assign previous displacements before computing new ones
                // -------------------------------------------------------
                for (int ir=0; ir<NRbins; ir++) {
                    prev_displ[ir] = displ[ir];
                }
                double costheta, sintheta;

                // Now we know how the utility varies as a function of the distance of detector i from the showers,
                // measured in terms of the position of the detector x[], y[]. We use this information to vary the
                // detector position by taking all detectors at the same radius and averaging the derivative.
                // ------------------------------------------------------------------------------------------------
                for (int id=0; id<Nunits; id++) {
                    if (keep_fixed[id]) continue;
                    double xi = x[id];
                    double yi = y[id];
                    double Ri; 
                    int ir; 
                    double dU_dRi;
                    double r2 = pow(xi,2.)+pow(yi,2.);
                    if (r2>0.) {
                        Ri = sqrt(r2);
                        costheta = xi/(Ri+epsilon);
                        sintheta = yi/(Ri+epsilon);
                        ir = (int)((Ri-rmin)/rspan*NRbins);
                        dU_dRi = dU_dxi[id]*costheta + dU_dyi[id]*sintheta;
                    } else {
                        dU_dRi = sqrt(pow(dU_dxi[id],2.)+pow(dU_dyi[id],2.));
                        ir = 0;
                    }
                    if (ir<NRbins) {
                        displ[ir] += k * f * dU_dRi * LearningRate[id]; // Learning rate is tied to detector unit
                        Ndispl[ir]++;
                    } else if (ir==NRbins) {
                        ir = NRbins-1;
                    } else {
                        cout    << "Warning, ir out of range" << endl;
                        outfile << "Warning, ir out of range" << endl;
                        warnings6++;
                    }
                } // End id loop

                // Compute average displacement as f(r)
                // ------------------------------------
                for (int ir=0; ir<NRbins; ir++) {
                    double R = rmin+(ir+0.5)*rspan/NRbins;
                    if (Ndispl[ir]>0) displ[ir] /= Ndispl[ir];

                    // Cap individual movements to be between 31% and 100% of maxdispl
                    // ---------------------------------------------------------------
                    float Current_mindispl = 0.31 * f * maxDispl; // We need to rescale the boundaries as the LR changes
                    float Current_maxdispl = 1.00 * f * maxDispl; 
                    int sgn_displ = 1;
                    if (displ[ir]<0.) sgn_displ = -1;
                    if (fabs(displ[ir])>Current_maxdispl) {
                        displ[ir] = Current_maxdispl*sgn_displ;
                    } else if (fabs(displ[ir])<Current_mindispl) {
                        displ[ir] = Current_mindispl*sgn_displ; 
                    }
                    // Protect from too rapid shrinkage (and compensate with similar bound in the other direction)
                    // -------------------------------------------------------------------------------------------
                    if (displ[ir]>0.5*R)  displ[ir] =  0.5*R;
                    if (displ[ir]<-0.5*R) displ[ir] = -0.5*R;
                    if (Ndispl[ir]>0) cout << ir << " " << Ndispl[ir] << " " << displ[ir] << endl;
                } // End ir loop

                // Now we have the required average displacement as a function of R and we apply to detectors
                // ------------------------------------------------------------------------------------------
                for (int id=0; id<Nunits; id++) {
                    if (keep_fixed[id]) continue;
                    double r2 = pow(x[id],2.)+pow(y[id],2.);
                    dx        = 0.;
                    dy        = 0.;
                    costheta  = 0.;
                    sintheta  = 0.;
                    double R  = 0.;
                    int ir    = 0;
                    dr2       = 0.;
                    dr2prev   = 0.;
                    if (r2>0.) { // Otherwise no movement
                        R = sqrt(r2);
                        ir = (int)((R-rmin)/rspan*NRbins);
                        costheta = x[id]/(R+epsilon);
                        sintheta = y[id]/(R+epsilon);
                        if (ir<NRbins) {
                            dx = costheta * displ[ir];
                            dy = sintheta * displ[ir];
                        }
                        if (PlotThis[10] || PlotThis[11]) {
                            dr2prev = pow(x[id]-xprev[id],2.)+pow(y[id]-yprev[id],2.);
                        }
                        // Ok, now update the positions
                        // ----------------------------
                        xprev[id] = x[id];
                        yprev[id] = y[id];
                        x[id] += dx;
                        y[id] += dy;
                    }
                    dr2 = pow(dx,2.) + pow(dy,2.); // Need this step as sqrt(0.) gives problems
                    if (dr2>0.) aveDispl += sqrt(dr2);     

                    // Accumulate information on how consistent are the movements of detectors
                    // -----------------------------------------------------------------------
                    if (epoch>0) CosThetaEff = ((x[id]-xprev[id])*dx + (y[id]-yprev[id])*dy)/
                                                (sqrt(pow(x[id]-xprev[id],2.)+pow(y[id]-yprev[id],2.)+epsilon)*sqrt(pow(dx,2.)+pow(dy,2.)+epsilon));

                    // Change LR based on the value
                    // of the angle between the current and the previous detector displacement. If positive,
                    // we increase the LR for that unit; if negative, we decrease it.
                    // -------------------------------------------------------------------------------------
                    double rate_modifier = -1.+2.*pow(0.5*(CosThetaEff+1.),2.); 
                                           // The above function is -1 for x=-1, +1 for x=1, and -0.3 for x=0

                    // Verify consistency of movements and modify learning rate for this radius
                    // ------------------------------------------------------------------------
                    LearningRate[id] *= exp(Momentum_coeff*rate_modifier); // This will apply to next iteration
                    // Clamp them - we do not want too much variation
                    // ----------------------------------------------
                    if (LearningRate[id]<MinLearningRate) LearningRate[id] = MinLearningRate;
                    if (LearningRate[id]>MaxLearningRate) LearningRate[id] = MaxLearningRate;

                    //if (!noplots) {
                    if (PlotThis[10] || PlotThis[11]) {
                        double ps = sqrt(dr2*dr2prev)/maxDispl2*CosThetaEff; // Renormalized scalar product
                        // Blow up region around zero to enhance that area in plot
                        ps = pow(fabs(ps),0.25);
                        if (CosThetaEff<0.) ps = -ps;
                        if (PlotThis[10]) {
                            if (epoch>0) {
                                // double ac = acos(CosThetaEff);
                                CosDir->Fill(ps); // CosDir->Fill(ac);
                            }
                        }
                        if (PlotThis[11]) {
                            if (epoch>0) {
                                // double ac = acos(CosThetaEff);
                                CosvsEp->Fill(epoch,ps); // CosvsEp->Fill(epoch,ac);
                            }
                        }
                    }
                    //}
#ifdef PLOTS
                    LR->Fill(epoch,log(LearningRate[id]));
#endif
                } // End id loop on dets

            } 
            if (CommonMode>=2) { // If MixedMode, some of the detectors can have been taken care of already in CM=0 part.

                // Detector array is divided in sets of n which share a
                // n-symmetry and whose displacements are the result
                // of averaging the n contributions
                // Beware - below we ASSUME the n-plets have same r, and phi
                // separated by 360/n degrees. This has to be enforced in routine
                // DefineLayout at the start of the program
                // --------------------------------------------------------------

                // Compute average displacement before equalization
                // ------------------------------------------------
                for (int im=0; im<Nmultiplets; im++) {
                    // Check that this multiplet does not have fixed units
                    // ---------------------------------------------------
                    if (keep_fixed[im]) continue;

                    // Check that this multiplet has not been broken earlier
                    // -----------------------------------------------------
                    int itrfirst = -1;
                    for (int itr=0; itr<multiplicity && itrfirst==-1; itr++) {
                        int ind = im+itr*Nmultiplets;
                        if (InMultiplet[ind]) itrfirst = itr;
                    }
                    if (itrfirst==-1) continue; // No elements can be moved together in this multiplet
                    double xi = x[im+itrfirst*Nmultiplets]; // Check: does indexing by multiplet make sense here?
                    double yi = y[im+itrfirst*Nmultiplets];
                    // Compute the starting angle of the first element of a multiplet
                    // --------------------------------------------------------------
                    double phifirst = PhiFromXY (xi,yi);

                    // Update detector positions in multiplets, based on avg gradient of U
                    // -------------------------------------------------------------------

                    // Decompose the movements into radial and azimuthal
                    // -------------------------------------------------
                    double avedp = 0.;
                    double avedr = 0.;
                    int ind;
                    double oldR, newR, oldPhi, newPhi, incr;
                    int NinMultiplet = 0;
                    for (int itr=itrfirst; itr<multiplicity; itr++) {
                        ind = im+itr*Nmultiplets;
                        if (InMultiplet[ind]) {
                            dx = k * f * dU_dxi[ind] * LearningRate[ind]; // LR is common to multiplet, but we keep it indexed by ind
                            dy = k * f * dU_dyi[ind] * LearningRate[ind];

                            // Cap individual movements to be between 31% and 100% of maxdispl
                            // ---------------------------------------------------------------
                            float Current_mindispl2 = 0.1*f*f*maxDispl2; // We need to rescale the boundaries as the LR changes
                            float Current_maxdispl2 = 1.0*f*f*maxDispl2; 
                            dr2 = dx*dx+dy*dy;
                            if (dr2>Current_maxdispl2) {
                                dx  = dx*sqrt(Current_maxdispl2/dr2);
                                dy  = dy*sqrt(Current_maxdispl2/dr2);
                                dr2 = Current_maxdispl2;
                            } else if (dr2<Current_mindispl2) {
                                dx  = dx*sqrt(Current_mindispl2/dr2);
                                dy  = dy*sqrt(Current_mindispl2/dr2);
                                dr2 = Current_mindispl2;
                            }

                            // See if detector wants to move farther from origin
                            // -------------------------------------------------
                            double xind = x[ind];
                            double yind = y[ind];
                            oldR = pow(xind,2.)+pow(yind,2.);
                            if (oldR>0.) {
                                oldR = sqrt(oldR);
                            } else { 
                                oldR = 0.;
                            }
                            newR = pow(xind+dx,2.)+pow(yind+dy,2.);
                            if (newR>0.) {
                                newR = sqrt(newR);
                            } else {
                                newR = 0.;
                            }
                            // if (fabs(newR-oldR)>maxDispl) cout << " for det " << ind << " newR, oldR are " << newR << " " << oldR << " dx,dy = " << dx << " " << dy << endl;
                            avedr += newR-oldR;

                            // We need to decompose the vector of displacements dx, dy into a displacement
                            // in radius, dr, and one in azimuth, dp. Rather than projecting the vector 
                            // (dx,dy) onto the radial direction and on the orthogonal direction, we just 
                            // take the angle corresponding to the position x+dx, y+dy for the three units
                            // ---------------------------------------------------------------------------
                            oldPhi = PhiFromXY (xind,yind);       // 0:2pi
                            newPhi = PhiFromXY (xind+dx,yind+dy); // 0:2pi
                            incr   = newPhi-oldPhi;
                            if (incr>pi) {
                                incr = 2.*pi-incr;
                            } else if (incr<-pi) {
                                incr = incr + 2.*pi;
                            }
                            avedp += incr;
                            NinMultiplet++;
                        } // If this ind was in multiplet
                    } // End loop on elements of a multiplet
                    if (NinMultiplet>0) {
                        avedr = avedr/NinMultiplet;
                        avedp = avedp/NinMultiplet;
                    }
                    double r = pow(xi,2.)+pow(yi,2.);
                    if (r>0.) {
                        r = sqrt(r);
                    } else {
                        r = 0.;
                    }

                    // Accumulate information on how consistent are the movements of detectors
                    // These are common displacements to the triplets now
                    // -----------------------------------------------------------------------
                    dx = (r+avedr)*cos(phifirst+avedp) - xi;
                    dy = (r+avedr)*sin(phifirst+avedp) - yi;
                    
                    // Update learning rate based on "costhetaeff" value (see above) - this is the cosine
                    // of the angle between the current and the previous detector displacement. If positive,
                    // we increase the LR for that unit; if negative, we decrease it.
                    // -------------------------------------------------------------------------------------
                    if (epoch>0) CosThetaEff = ((xi-xprev[im])*dx + (yi-yprev[im])*dy)/
                                               (sqrt(pow(xi-xprev[im],2)+pow(yi-yprev[im],2)+epsilon)*sqrt(pow(dx,2)+pow(dy,2)+epsilon));
                    double rate_modifier = CosThetaEff; // If using an average one, it might be better to instead use the following definition: 
                                                        // rate_modifier = -1.+2.*pow(0.5*(CosThetaEff+1.),2); 
                                                        // the above function is -1 for x=-1, +1 for x=1, and -0.3 for x=0

                    // Ok, now update the positions of all members of a multiplet
                    // ----------------------------------------------------------
                    for (int itr=itrfirst; itr<multiplicity; itr++) {
                        int ind = im+itr*Nmultiplets;
                        if (InMultiplet[ind]) {
                            if (PlotThis[10] || PlotThis[11]) dr2prev = pow(x[ind]-xprev[ind],2.)+pow(y[ind]-yprev[ind],2.);
                            xprev[ind]   = x[ind];
                            yprev[ind]   = y[ind];
                            x[ind]       = (r+avedr)*cos(phifirst+avedp+2.*pi*(itr-itrfirst)/multiplicity);
                            y[ind]       = (r+avedr)*sin(phifirst+avedp+2.*pi*(itr-itrfirst)/multiplicity);
                            if (PlotThis[10] || PlotThis[11]) dr2 = pow(x[ind]-xprev[ind],2.)+pow(y[ind]-yprev[ind],2.);

                            // Account for displacement
                            // Note, we add it up here per unit instead of doing it outside of this loop
                            // once per multiplet, as there are potentially missing units in the multiplet
                            // ---------------------------------------------------------------------------
                            if (dr2>0.) aveDispl += sqrt(dr2);
                            
                            LearningRate[ind] *= exp(Momentum_coeff*rate_modifier);
                            // Clamp them - we do not want too much variation
                            if (LearningRate[ind]<MinLearningRate) LearningRate[ind] = MinLearningRate;
                            if (LearningRate[ind]>MaxLearningRate) LearningRate[ind] = MaxLearningRate;
#ifdef PLOTS
                            LR->Fill(epoch,log(LearningRate[ind]));
#endif
                            //if (!noplots) {
                            if (PlotThis[10] || PlotThis[11]) {
                                double ps = sqrt(dr2*dr2prev)/maxDispl2*CosThetaEff; // Renormalized scalar product
                                // Blow up region around zero to enhance that area in plot
                                ps = pow(fabs(ps),0.25);
                                if (CosThetaEff<0.) ps = -ps;
                                if (PlotThis[10]) {
                                    if (epoch>0) {
                                        // double ac = acos(CosThetaEff);
                                        CosDir->Fill(ps); // CosDir->Fill(ac);
                                    }
                                } 
                                if (PlotThis[11]) {
                                    if (epoch>0) {
                                        // double ac = acos(CosThetaEff);
                                        CosvsEp->Fill(epoch,ps); // CosvsEp->Fill(epoch,ac);
                                    }
                                }
                            }
                            //}
                        }
                    }

                    // cout << "     ID = " << im << ": x+dx,y+dy = " << x[im];
                    // if (dx>=0.) cout << "+";
                    // cout  << dx << "," << y[im];
                    // if (dy>=0.) cout << "+";
                    // cout << dy << endl;
                    if (scanU) {
                        cout << "xprev, x, dudx = " << xprev[idstar] << " " << x[idstar] << " " << dU_dxi[idstar] 
                            << " yprev, y, dudy = " << yprev[idstar] << " " << y[idstar] << " " << dU_dyi[idstar] << endl;
                    } 
                } // End im loop on multiplets
            } // End CommonMode if block

            // Renormalize average distance 
            // ----------------------------
            aveDispl /= Nunits;
            
            // Recenter array before enforcing forbidden region boundaries
            // -----------------------------------------------------------
            if (KeepCentered && CommonMode==0) RecenterArray();

            // Enforce forbidden region
            // ------------------------
            if (VoidRegion) {
                //IncludeOnlyFR = false;
                //FRpar[0] = 0.5;
                //FRpar[1] = 200.;
                //ForbiddenRegion (1,0,0); // in y region below line y=0.5x+200
                //IncludeOnlyFR = false;
                //FRpar[0] = -100.;
                //FRpar[1] = 300.;
                //ForbiddenRegion (2,0,0); // in x region [-100,300] 
                //IncludeOnlyFR = true;
                //FRpar[0] = 0.;
                //FRpar[1] = 0.;
                //FRpar[2] = 1200.;
                //ForbiddenRegion(0,0,0); // within a 1200 m circle

                // Pampalabola: triangle at positions
                // A = (-3499.,2000.) B = (1800,2000) C = (1800,-3508). BAC is -0.804735, intercept -816
                // ---------------------
                IncludeOnlyFR = false;     // Allowed band 
                FRpar[0] = -3499.;         // between x=-3499
                FRpar[1] = 1800.;          // ...and x=1800
                ForbiddenRegion(2,0,0);    // Semiplane vertically aligned

                IncludeOnlyFR = true;      // Allowed region is above line
                FRpar[0] = -tan(0.804735); // Negative slope
                FRpar[1] = -1637.;         // Intercept (0,-1637)
                ForbiddenRegion(1,0,0);    // Semiplane not containing origin, not vertically aligned

                IncludeOnlyFR = false;     // Allowed band
                FRpar[0] = 0.;             // Slope = 0 (horizontal)
                FRpar[1] = 2000.;          // Below y=2000
                ForbiddenRegion(1,0,0);    // Semiplane not containing origin, not vertically aligned
            }

            // Handle overlap of detector units
            // --------------------------------
            if (epoch%50==0 || epoch==maxEpochs-1) ResolveOverlaps(); // To speed things up, we could also do this only once every 10 epochs or only at the end...
        } // End if !noSGDupdate
        // Comment through to here if you want to bypass SGD (for fixed config runs): */

        cout    << "     Epoch = " << epoch 
                << "  Utility value = " << Utility <<  " aveDispl = " << aveDispl;
        outfile << "     Epoch = " << epoch 
                << "  Utility value = " << Utility << " aveDispl = " << aveDispl;
        if (CommonMode==0) {
            cout    << " avedU = " << avedUdR << " Exposure = " << ExposureFactor;
            outfile << " avedU = " << avedUdR << " Exposure = " << ExposureFactor;
        } else if (CommonMode==1) {
            cout    << " Exposure = " << ExposureFactor << endl;
            outfile << " Exposure = " << ExposureFactor << endl;
        } else if (CommonMode>=2) {
            cout    << " avedU = " << avedUdR  << " Exposure = " << ExposureFactor;
            outfile << " avedU = " << avedUdR  << " Exposure = " << ExposureFactor;            
        }
        //cout << " aver ratio dedr = " << sumrat/nsum << endl;
        cout    << endl;
        outfile << endl;
        if (CheckInitialization) {
            cout << "     " << Start_true_wins << " Good initialization over " << Start_true_trials << endl;
            cout << endl;
        }

        // Histogram of R distribution
        // ---------------------------
        //if (!noplots) {
            if (PlotThis[2]) {
                for (int id=0; id<Nunits; id++) {
                    double r = sqrt(x[id]*x[id]+y[id]*y[id]);
                    //if (r==0) r = 1.;
                    Rdistr->Fill(r); //,1./(twopi*r));
                }
                // Ensure the Rdistribution histograms stays visible
                // -------------------------------------------------
                float hmax = 0;
                for (int ib=0; ib<NbinsRdistr; ib++) {
                    float h = Rdistr->GetBinContent(ib+1);
                    if (h>hmax) hmax = h;
                    h = Rdistr0->GetBinContent(ib+1);
                    if (h>hmax) hmax = h;
                }
                Rdistr0->SetMaximum(hmax*1.1);
            }
        //} // End !noplots
        // Histogram of Phi distribution
        // -----------------------------
        /*if (PlotThis[11]) {
            for (int id=0; id<Nunits; id++) {
                Pdistr->Fill(PhiFromXY(x[id],y[id]));
            }
            // Ensure the Pdistribution histograms stays visible
            // -------------------------------------------------
            int hmax = 0;
            for (int ib=0; ib<NbinsRdistr; ib++) {
                int h = Pdistr->GetBinContent(ib+1);
                if (h>hmax) hmax = h;
                h = Pdistr0->GetBinContent(ib+1);
                if (h>hmax) hmax = h;
            }
            Pdistr0->SetMaximum(hmax*1.1);
        }*/
#ifdef PLOTS
        // Compute agreement metric
        double QP = 0.5*sqrt(pow(DXP->GetRMS(),2)+pow(DYP->GetRMS(),2)+pow(DXG->GetRMS(),2)+pow(DYG->GetRMS(),2));
        double QA = 0.5*sqrt(pow(DTHG->GetRMS(),2)+pow(DTHP->GetRMS(),2)+pow(DPHG->GetRMS(),2)+pow(DPHP->GetRMS(),2));
        double QE = DEG->GetMean(); // 0.5*sqrt(pow(DEG->GetRMS(),2)+pow(DEP->GetRMS(),2));
        double chi2 = 0.;
        double cumt = 0;
        double cumm = 0;
        double dmax = 0.;
        double sumt = HEtrue->GetEntries();
        double summ = HEmeas->GetEntries();
        for (int i=1; i<=10; i++) {
            int N_t = HEtrue->GetBinContent(i);
            int N_m = HEmeas->GetBinContent(i);
            cumt += 1.*N_t/sumt;
            cumm += 1.*N_m/summ;
            if (dmax<fabs(cumt-cumm)) dmax = fabs(cumt-cumm);
            if (N_t>0) chi2 += pow(1.*(N_t-N_m)/N_t,2.);
        }
        chi2 = chi2 / 10.;
        cout    << "     Performance metric of shower position likelihood = " << QP << " " << QA << " " << QE << " " 
                << chi2 << " " << dmax << " avg steps = " << 1.*NumAvgSteps/DenAvgSteps << endl;
        outfile << "     Performance metric of shower position likelihood = " << QP << " " << QA << " " << QE << " " 
                << chi2 << " " << dmax << " avg steps = " << 1.*NumAvgSteps/DenAvgSteps << endl;
        PosQ->Fill(epoch,QP);
        AngQ->Fill(epoch,QA);
        EQ->Fill(epoch,QE*100.);

        // Current distances plot
        // ----------------------
        if (C1!=nullptr) {
            delete C1;
        }
        C1 = new TCanvas ("C1","",1000,500);
        C1->Divide(5,2);
        C1->cd(1);
        DXP->Draw();
        C1->cd(2);
        DYP->Draw();
        C1->cd(3);
        DXG->Draw();
        C1->cd(4);
        DYG->Draw();
        C1->cd(5);
        DTHP->Draw();
        C1->cd(6);
        DPHP->Draw();
        C1->cd(7);
        DTHG->Draw();
        C1->cd(8);
        DPHG->Draw();
        C1->cd(9);
        DTHPvsT->Draw("COLZ");
        C1->cd(10);
        DTHGvsT->Draw("COLZ");
        C1->Update();
#endif

        //if (!noplots) {

            // Summary plot
            // ------------
            if (Nplots_CT>0) {
                int nx = 2;
                int ny = 1;
                float spanx = 300;
                float spany = 300;
                if (Nplots_CT==2) {
                    spanx = 700;
                    spany = 400;
                    nx = 2;
                    ny = 1;
                } else if (Nplots_CT==3) {
                    spanx = 900;
                    spany = 300;
                    nx = 3;
                    ny = 1;                
                } else if (Nplots_CT==4) {
                    spanx = 700;
                    spany = 700;
                    nx = 2;
                    ny = 2;
                } else if (Nplots_CT<7) {
                    spanx = 1100;
                    spany = 700;
                    nx = 3;
                    ny = 2;
                } else if (Nplots_CT<9) {
                    spanx = 1400;
                    spany = 700;
                    nx = 4;
                    ny = 2;
                } else if (Nplots_CT<10) {
                    spanx = 900;
                    spany = 900;
                    nx = 3;
                    ny = 3;
                } else if (Nplots_CT<11) {
                    spanx = 1500;
                    spany = 600;
                    nx = 5;
                    ny = 2;
                } else if (Nplots_CT<13) {
                    spanx = 1200;
                    spany = 900;
                    nx = 4;
                    ny = 3;
                } else if (Nplots_CT<16) {
                    spanx = 1500;
                    spany = 900;
                    nx = 5;
                    ny = 3;
                } else {
                    spanx = 1000;
                    spany = 1000;
                    nx = 4;
                    ny = 4;
                }
                int ipanel = 1;
                if (CT != nullptr) {
                    delete CT;
                }
                CT = new TCanvas ("CT","",spanx,spany);
                if (Nplots_CT==1) {
                    CT->cd();
                } else {
                    CT->Divide(nx,ny);
                }
                if (PlotThis[0]) {
                    CT->cd(ipanel);
                    //if (epoch>0) U->Fit("pol1","Q");
                    U->Draw("P");
                    Uave->Draw("SAME");
                    ipanel++;
                }
                if (PlotThis[1]) {
                    CT->cd(ipanel);
                    //CT->Range(-TotalRspan,-TotalRspan,TotalRspan,TotalRspan);
                    //Pampalabola->Draw();
                    if (plotdensity) {
                        //Showers3->SetMarkerColorAlpha(kBlue,0.1);
                        Showers3->Draw("COLZ");
                    } else {
                        //Showers3p->SetMarkerColorAlpha(kBlue,0.1);
                        Showers3p->Draw("COLZ");
                    }
                    //Layout->SetMarkerColorAlpha(kRed,0.3);
                    Layout->Draw("PSAME");
                    // Draw here the boundaries of the Pampalabola site
                    // ------------------------------------------------
                    double leftboundary = -3499.;
                    if (leftboundary<-TotalRspan) leftboundary = -TotalRspan;
                    double rightboundary = 1800.;
                    if (rightboundary>TotalRspan) rightboundary = TotalRspan;
                    double topboundary = 2000.;
                    if (topboundary>TotalRspan) topboundary = TotalRspan;
                    double bottomboundary = -3508.;
                    if (bottomboundary<-TotalRspan) bottomboundary = -TotalRspan;
                    if (VoidRegion) {
                        // Horizontal line at the top
                        Line1 = new TLine (leftboundary,topboundary,rightboundary,topboundary);
                        Line1->Draw();
                        // Vertical boundary to the right
                        Line2 = new TLine (rightboundary,topboundary,rightboundary,bottomboundary);
                        Line2->Draw();
                        // Diagonal line - here we must first find the intercepts of the line
                        // With the plot boundaries
                        // ------------------------------------------------------------------
                        // (x1,y1) = (-3499,2000) 
                        // (x2,y2) = (1800,-3508)
                        // y = y1 -x1*(y2-y1)/(x2-x1) +  (y2-y1)/(x2-x1) * x  
                        // that is 
                        //         m = (y2-y1)/(x2-x1)
                        //         q = y1-x1*m
                        double x1 = -3499;
                        double x2 = 1800;
                        double y1 = 2000;
                        double y2 = -3508;
                        double m  = (y2-y1)/(x2-x1);
                        double q  = y1-x1*m;
                        double x_miny = (-TotalRspan-q)/m;
                        double y_minx = -TotalRspan*m + q;
                        double y_miny = -TotalRspan;
                        double x_minx = -TotalRspan;
                        if (x_miny>x2) {
                            x_miny = x2;
                            y_miny = y2;
                        }
                        if (y_minx>y1) {
                            y_minx = y1;
                            x_minx = x1;
                        }  
                        Line3 = new TLine (x_minx, y_minx, x_miny, y_miny);
                        Line3->Draw();
                    }
                    ipanel++;
                }
                if (PlotThis[2]) {
                    CT->cd(ipanel);
                    Rdistr0->Draw();
                    Rdistr->Draw("SAME");
                    ipanel++;
                }
                if (PlotThis[3]) {
                    CT->cd(ipanel);
                    float UaveMin = 0.8;
                    float UaveMax = 1.2;
                    for (int ib=1; ib<=NbinsProfU; ib++) {
                        double content = UaveGF->GetBinContent(ib);
                        if (content<UaveMin) UaveMin = content; 
                        if (content>UaveMax) UaveMax = content; 
                        content = UaveIR->GetBinContent(ib);
                        if (content<UaveMin) UaveMin = content; 
                        if (content>UaveMax) UaveMax = content; 
                        content = UavePR->GetBinContent(ib);
                        if (content<UaveMin) UaveMin = content; 
                        if (content>UaveMax) UaveMax = content; 
                        //content = UaveTC->GetBinContent(ib);
                        //if (content<UaveMin) UaveMin = content; 
                        //if (content>UaveMax) UaveMax = content; 
                    }
                    if (eta_GF>0. || PeVSource) {
                        UaveGF->SetMinimum(UaveMin-0.05);
                        UaveGF->SetMaximum(UaveMax+0.05);
                        UaveGF->Draw("PE");
                    }
                    if (eta_IR>0. || PeVSource) {
                        if (eta_GF==0. && !PeVSource) {
                            UaveIR->SetMinimum(UaveMin-0.05);
                            UaveIR->SetMaximum(UaveMax+0.05);
                            UaveIR->Draw("PE");
                        } else {
                            UaveIR->Draw("SAME");
                        }
                    }
                    if (eta_PR>0. || PeVSource) {
                        if (eta_GF==0 && eta_IR==0. && !PeVSource) {
                            UavePR->SetMinimum(UaveMin-0.05);
                            UavePR->SetMaximum(UaveMax+0.05);
                            UavePR->Draw("PE");
                        } else {
                            UavePR->Draw("SAME");
                        }
                    }
                    if (eta_TA>0. || eta_TL>0.) {
                        UaveTC->Draw("SAME");
                    }
                    ipanel++;
                }
                if (PlotThis[4]) {
                    CT->cd(ipanel);
                    for (int ib=1; ib<=NbinsProfU; ib++) {
                        double content;
                        if (epoch==0) {
                            content = DR0->GetBinContent(ib);
                            if (content!=0.) {
                                if (content<dRMin) dRMin = content; 
                                if (content>dRMax) dRMax = content;
                            }
                        } 
                        content = DR->GetBinContent(ib);
                        if (content!=0.) {
                            if (content<dRMin) dRMin = content; 
                            if (content>dRMax) dRMax = content; 
                        }
                    }
                    DR0->SetMinimum(dRMin-0.2);
                    DR0->SetMaximum(dRMax+0.2);
                    DR0->Draw();
                    DR->Draw("SAME");
                    ipanel++;
                }
                if (PlotThis[5]) {
                    CT->cd(ipanel);
                    CT->GetPad(ipanel)->SetLogy();
                    // LR->Draw();
                    DE0->Draw();
                    DE->Draw("SAME");
                    ipanel++;
                }
                if (PlotThis[6]) {
                    CT->cd(ipanel);
                    Pareto->Draw();
                    //CT->GetPad(ipanel)->SetLogy();
                    //LLRG->Draw();
                    //LLRP->Draw("SAME");
                    ipanel++;
                }
                if (PlotThis[7]) {
                    CT->cd(ipanel);
                    double duMax = 1.2;
                    for (int ib=1; ib<=75; ib++) {
                        double content = DUGF->GetBinContent(ib);
                        if (content>duMax) duMax = content; 
                        content = DUIR->GetBinContent(ib);
                        if (content>duMax) duMax = content; 
                        content = DUPR->GetBinContent(ib);
                        if (content>duMax) duMax = content; 
                        content = DUTC->GetBinContent(ib);
                        if (content>duMax) duMax = content; 
                    }
                    if (eta_GF>0.) {
                        DUGF->SetMaximum(duMax*1.2);
                        DUGF->Draw("");
                    }
                    if (eta_IR>0.) {
                        if (eta_GF==0.) {
                            DUIR->SetMaximum(duMax*1.2);
                            DUIR->Draw("");
                        } else {
                            DUIR->Draw("SAME");
                        }
                    }
                    if (eta_PR>0.) {
                        if (eta_GF*eta_IR==0.) {
                            DUPR->SetMaximum(duMax*1.2);
                            DUPR->Draw("");
                        } else {
                            DUPR->Draw("SAME");
                        }
                    }
                    if (eta_TA>0. || eta_TL>0.) {
                        if (eta_GF*eta_IR*eta_PR==0.) {
                            DUTC->SetMaximum(duMax*1.2);
                            DUTC->Draw("");
                        } else {
                            DUTC->Draw("SAME");
                        }
                    }
                    ipanel++;
                }
                if (PlotThis[8]) {
                    CT->cd(ipanel);
                    // DR0vsE->Draw();
                    ipanel++;
                }
                if (PlotThis[9]) {
                    double hmax = 0.;
                    for (int i=1; i<=10; i++) {
                        double c = HEtrue->GetBinContent(i);
                        if (hmax<c) hmax=c;
                        c = HEmeas->GetBinContent(i);
                        if (hmax<c) hmax=c;
                    }
                    HEtrue->SetMaximum(hmax*1.2);
                    CT->cd(ipanel);
                    HEtrue->Draw();
                    HEmeas->Draw("SAME");
                    ipanel++;
                }
                if (PlotThis[10]) {
                    CT->cd(ipanel);
                    CosDir->Draw();
                    ipanel++;
                }
                if (PlotThis[11]) {
                    CT->cd(ipanel);
                    CosvsEp->Draw();
                    ipanel++;
                }
                if (PlotThis[12]) {
                    CT->cd(ipanel);
                    Layout2->Draw();
                    //Pdistr0->SetMinimum(0.);
                    //Pdistr0->Draw();
                    //Pdistr->Draw("SAME");
                    ipanel++;
                }
                if (PlotThis[13]) {
                    CT->cd(ipanel);
                    ThGIvsThGP->Draw("COLZ");
                    //numx->Draw();
                    //numy->Draw("SAME");
                    ipanel++;
                }
                if (PlotThis[14]) {
                    CT->cd(ipanel);
                    LrGIvsLrGP->Draw("COLZ");
                    //denx->Draw();
                    //deny->Draw("SAME");
                    ipanel++;
                }
                if (PlotThis[15]) {
                    CT->cd(ipanel);
                    dUdR->Draw();
                }
                CT->Update();

                char namepng[120];
#ifdef STANDALONE
                sprintf (namepng, "/lustre/cmswork/dorigo/swgo/MT/Layouts/Layout_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, startEpoch+epoch+1);
#endif
#ifdef UBUNTU
                sprintf (namepng, "/home/tommaso/Work/swgo/MT/Layouts/Layout_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, startEpoch+epoch+1);
#endif
#ifdef INROOT
                sprintf (namepng, "./SWGO/Layouts/Layout_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, startEpoch+epoch+1);
#endif
                CT->Print(namepng);
            } // End if nplots_ct>0
        //} // End if !noplots

#ifdef PLOTS
        TCanvas * CT2 = new TCanvas ("CT2","",1000,300);
        CT2->Divide(3,1);
        CT2->cd(1);
        PosQ->Draw();
        CT2->cd(2);
        LR->Draw();
        CT2->cd(3);
        double tmp;
        double maxh = -largenumber;
        for (int i=1; i<=epoch+1; i++) {
            tmp = U_gf->GetBinContent(i);
            if (tmp>maxh) maxh = tmp;
            tmp = U_ir->GetBinContent(i);
            if (tmp>maxh) maxh = tmp;
            tmp = U_pr->GetBinContent(i);
            if (tmp>maxh) maxh = tmp;
            tmp = U_tc->GetBinContent(i);
            if (tmp>maxh) maxh = tmp;
        }
        U_gf->SetMinimum(0.);
        U_gf->SetMaximum(maxh+0.1*maxh);
        U_gf->Draw();
        U_ir->Draw("SAME");
        U_pr->Draw("SAME");
        U_tc->Draw("SAME");
#endif

        // Adjust too small learning rates here
        // ------------------------------------
        //if (aveDispl<maxDispl/10.) LearningRate = LearningRate*1.5;


        // Debug: check value of utility for same generated batch, after coordinates update
        // --------------------------------------------------------------------------------
        double OldUtility = Utility;
        int iter = 0;
        if (checkUtility && epoch>0) {
            do {
                Ng_active = 0;
                Np_active = 0;
                for (int is=0; is<Nevents+Nbatch; is++) {
                    logLRT[is]   = 0.;
                    sigmaLRT[is] = 1.;
                    double p = myRNG->Uniform();
                    if ((is<Nevents && is%2==0) || p<GenGammaFrac) {
                        IsGamma[is] = true;
                    } else {
                        IsGamma[is] = false;
                    }
                    GenerateShower (is);
                    bool ShowerOK = FindLogLR (is); // Fills logLRT[] array and sigmaLRT
                    if (!ShowerOK) {
                        Active[is]  = false;
                        PActive[is] = 0.;
                        continue;
                    }
                    if (is>=Nevents) {
                        if (IsGamma[is]) {
                            Ng_active += PActive[is];
                        } else {
                            Np_active += PActive[is];
                        }
                    }
                } // end is loop
                N_active = Ng_active+Np_active;
                TrueGammaFraction = Ng_active/N_active; // For batch events only

                // Compute the PDF of the test statistic for all batch showers
                // -----------------------------------------------------------
                for (int k=Nevents; k<Nevents+Nbatch; k++) {
                    if (!Active[k]) continue;
                    pg[k] = ComputePDF (k,true);
                    pp[k] = ComputePDF (k,false);
                }

                MeasFg = MeasuredGammaFraction (); // Also computes static inv_sigmafs2
                if (inv_sigmafs2==0.) {
                    inv_sigmafs2 = epsilon;
                    cout    << "Warning, inv_sigmafs2 = 0" << endl;
                    outfile << "Warning, inf_sigmafs2 = 0" << endl;
                    warnings1++;
                    TerminateAbnormally ();
                    return 0;            
                }
                sigmafs2 = 1./inv_sigmafs2;
                MeasFgErr = sqrt(sigmafs2);
                inv_sigmafs  = 1./MeasFgErr; // Above we were computing the variance with RCF bound
                cout << endl;
                cout    << "     GF = " << MeasFg << " +- " << MeasFgErr << " (true = " << TrueGammaFraction << ")"; 
                outfile << "     GF = " << MeasFg << " +- " << MeasFgErr << " (true = " << TrueGammaFraction << ")"; 
                if (!usetrueXY)   ComputeUtilityGF();
                if (!usetrueE)    ComputeUtilityIR();
                if (!usetrueAngs) ComputeUtilityPR();
                if (PeVSource) {
                    ComputeUtilityPeVSource (); // Utility is written in U_GF
                } 
                double U_TC = 0.;
                if (UseAreaCost) {
                    ComputeUtilityArea(); // Computes U_TA if default mode=0
                    U_TC += U_TA;
                }
                if (UseLengthCost) {
                    ComputeUtilityLength();
                    U_TC += U_TL;
                }
                Utility = 0.;
                if (PeVSource) {
                    Utility = U_PS; // + U_TC;  // Leave U_TC out of total utility, only use its derivative
                } else {
                    if (eta_GF>0.) Utility += U_GF;
                    if (eta_IR>0.) Utility += U_IR;
                    if (eta_PR>0.) Utility += U_PR;
                    // if (eta_TL + eta_TA>0.) Utility += U_TC; // Leave U_TC out of total utility, only use its derivative
                }
                cout << " New vs Old U = " << Utility << " " << OldUtility;
                if (Utility<OldUtility) {
                    for (int id=0; id<Nunits; id++) {
                        x[id] = 0.5*x[id]+0.5*xprev[id];
                        y[id] = 0.5*y[id]+0.5*yprev[id];
                    }
                    cout << " - halving displacements" << endl;
                    iter++;
                } else {
                    cout << " - moving on" << endl;
                }
            } while (Utility<OldUtility && iter<15);
        }

        // Check that everything is in order
        // ---------------------------------
        if (warnings1+warnings2+warnings3!=0) {
            TerminateAbnormally ();
            // SaveLayout();
            return 0;
        }

        // New epoch coming
        // ----------------
        epoch++;

#ifdef PLOTS
        // if (C1!=nullptr) {
        //    delete C1;
        //    delete CT2;
        // }
#endif
        // if (CT!=nullptr) {
        //     delete CT;
        // }
    } while (epoch<Nepochs); // end SGD loop
    // -------------------------------------
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef FEWPLOTS
    if (scanU) {
        C0 = new TCanvas ("C0","",1600,500);
        C0->Divide(3,1);
        C0->cd(1);
        Uvsxy->Draw("COLZ");
        C0->cd(2);
        Uvsx->Draw();
        C0->cd(3);
        Uvsy->Draw();
        C0->Update();
        return 0;
    }
#endif

    TCanvas * ddr = new TCanvas ("ddr","",700,700);
    ddr->Divide(2,5);
    for (int i=0; i<10; i++) {
        ddr->cd(i+1);
        dEdR[i]->Draw();
    }

    // Control graph of LogLR approximation
    // ------------------------------------
    TCanvas * st = new TCanvas ("st","", 500,500);
    if (SampleT) {
        st->cd();
        SvsS->Draw();
        SvsSP->Draw("SAME");
    }

    // Report mean utility after Nepochs loop, if noSGDupdate is on
    // ------------------------------------------------------------
    if (noSGDupdate) {
        if (Nepochs==0) Nepochs = 1;
        sumUGF = sumUGF/Nepochs;
        sumUIR = sumUIR/Nepochs;
        sumUPR = sumUPR/Nepochs;
        sumUTC = sumUTC/Nepochs;
        sumUPS = sumUPS/Nepochs;
        double rmsmGF = sqrt(sumUGF2/Nepochs-pow(sumUGF,2.));
        if (Nepochs>1) rmsmGF /= sqrt(Nepochs-1.);
        double rmsmIR = sqrt(sumUIR2/Nepochs-pow(sumUIR,2.));
        if (Nepochs>1) rmsmIR /= sqrt(Nepochs-1.);
        double rmsmPR = sqrt(sumUPR2/Nepochs-pow(sumUPR,2.));
        if (Nepochs>1) rmsmPR /= sqrt(Nepochs-1.);
        double rmsmTC = sqrt(sumUTC2/Nepochs-pow(sumUTC,2.));
        if (Nepochs>1) rmsmTC /= sqrt(Nepochs-1.);
        double rmsmPS = sqrt(sumUPS2/Nepochs-pow(sumUPS,2.));
        if (Nepochs>1) rmsmPS /= sqrt(Nepochs-1.);
        cout    << endl;
        cout    << "     Average utility after " << Nepochs << " epochs: " << endl;
        cout    << "       U_GF = " << sumUGF << " +- " << rmsmGF << endl;
        cout    << "       U_IR = " << sumUIR << " +- " << rmsmIR << endl;
        cout    << "       U_PR = " << sumUPR << " +- " << rmsmPR << endl;
        cout    << "       U_TC = " << sumUTC << " +- " << rmsmTC << endl;
        cout    << "       U_PS = " << sumUPS << " +- " << rmsmPS << endl;
        cout    << "       U_1  = " << sumUGF+sumUIR+sumUPR+sumUTC << " +- " << sqrt(rmsmGF*rmsmGF+rmsmIR*rmsmIR+rmsmPR*rmsmPR+rmsmTC*rmsmTC) << endl;
        cout    << endl;
        outfile << endl;
        outfile << "     Average utility after " << Nepochs << " epochs: " << endl;
        outfile << "       U_GF = " << sumUGF << " +- " << rmsmGF << endl;
        outfile << "       U_IR = " << sumUIR << " +- " << rmsmIR << endl;
        outfile << "       U_PR = " << sumUPR << " +- " << rmsmPR << endl;
        outfile << "       U_PR = " << sumUTC << " +- " << rmsmTC << endl;
        outfile << "       U_PS = " << sumUPS << " +- " << rmsmPS << endl;
        outfile << "       U_1  = " << sumUGF+sumUIR+sumUPR+sumUTC << " +- " << sqrt(rmsmGF*rmsmGF+rmsmIR*rmsmIR+rmsmPR*rmsmPR+rmsmTC*rmsmTC) << endl;
        outfile << endl;
    } else {

        // Get min and max utility from profile graph
        // ------------------------------------------
        cout << endl;
        cout << "     Initial utility:" << endl;
        cout << "     U    = " << Uave->GetBinContent(1)   << " +- " << Uave->GetBinError(1) << endl;
        cout << "     Final utility:" << endl;
        cout << "     U    = " << Uave->GetBinContent(NbinsProfU)   << " +- " << Uave->GetBinError(NbinsProfU) << endl;
        cout << "     U_GF ratio = " << UaveGF->GetBinContent(NbinsProfU) << " +- " << UaveGF->GetBinError(NbinsProfU) << endl;
        cout << "     U_IR ratio = " << UaveIR->GetBinContent(NbinsProfU) << " +- " << UaveIR->GetBinError(NbinsProfU) << endl;
        cout << "     U_PR ratio = " << UavePR->GetBinContent(NbinsProfU) << " +- " << UavePR->GetBinError(NbinsProfU) << endl;
        cout << "     U_TC ratio = " << UaveTC->GetBinContent(NbinsProfU) << " +- " << UaveTC->GetBinError(NbinsProfU)  << endl;
        cout << endl;
        outfile << endl;
        outfile << "     Initial utility:" << endl;
        outfile << "     U    = " << Uave->GetBinContent(1)   << " +- " << Uave->GetBinError(1) << endl;
        outfile << "     Final utility:" << endl;
        outfile << "     U    = " << Uave->GetBinContent(NbinsProfU)   << " +- " << Uave->GetBinError(NbinsProfU) << endl;
        outfile << "     U_GF ratio = " << UaveGF->GetBinContent(NbinsProfU) << " +- " << UaveGF->GetBinError(NbinsProfU) << endl;
        outfile << "     U_IR ratio = " << UaveIR->GetBinContent(NbinsProfU) << " +- " << UaveIR->GetBinError(NbinsProfU) << endl;
        outfile << "     U_PR ratio = " << UavePR->GetBinContent(NbinsProfU) << " +- " << UavePR->GetBinError(NbinsProfU) << endl;
        outfile << "     U_TC ratio = " << UaveTC->GetBinContent(NbinsProfU) << " +- " << UaveTC->GetBinError(NbinsProfU)  << endl;
        outfile << endl;
    }

#ifdef PLOTRESOLUTIONS
    // Plot Histograms of resolutions for paper v2
    // -------------------------------------------
    for (int ibin=0; ibin<20; ibin++) {
        // We do not rely on single estimates, remove them
        // -----------------------------------------------
        if (DE0vsE->GetBinEntries(ibin)<1) DE0vsE->SetBinContent(ibin,0.);
        if (DE0vsR->GetBinEntries(ibin)<1) DE0vsR->SetBinContent(ibin,0.);
        if (DR0vsE->GetBinEntries(ibin)<1) DR0vsE->SetBinContent(ibin,0.);
        if (DR0vsR->GetBinEntries(ibin)<1) DR0vsR->SetBinContent(ibin,0.);
        for (int jbin=0; jbin<20; jbin++) {
            int bin = DE0vsER->GetBin(ibin,jbin);
            if (DE0vsER->GetBinEntries(bin)<1) {
                DE0vsER->SetBinContent(ibin,jbin,0.);
            }
            bin = DR0vsER->GetBin(ibin,jbin);
            if (DR0vsER->GetBinEntries(bin)<1) {
                DR0vsER->SetBinContent(ibin,jbin,0.);
            }
        }
    }
    DE0vsE->SetMinimum(0.);
    DE0vsR->SetMinimum(0.);
    DR0vsE->SetMinimum(0.);
    DR0vsR->SetMinimum(0.);
    TCanvas * Resoe = new TCanvas ("Resoe", "", 1400,500);
    Resoe->Divide(3,1);
    Resoe->cd(1);
    DE0vsER->Draw("COLZ");
    Resoe->cd(2);
    DE0vsE->Draw("PE");
    Resoe->cd(3);
    DE0vsR->Draw("PE");
 #ifdef STANDALONE
    sprintf (namepng,"/lustre/cmswork/dorigo/swgo/MT/Plots/Resolutions_E_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, epoch+startEpoch);
 #endif
 #ifdef UBUNTU
    sprintf (namepng,"/home/tommaso/Work/swgo/MT/Plots/Resolutions_E_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, epoch+startEpoch);
 #endif
 #ifdef INROOT
    sprintf (namepng,"./SWGO/Root/Resolutions_E_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, epoch+startEpoch);
 #endif
    Resoe->Print(namepng);
    TCanvas * Resor = new TCanvas ("Resor", "", 1400,500);
    Resor->Divide(3,1);
    Resor->cd(1);
    DR0vsER->Draw("COLZ");
    Resor->cd(2);
    DR0vsE->Draw("PE");
    Resor->cd(3);
    DR0vsR->Draw("PE");
 #ifdef STANDALONE
    sprintf (namepng,"/lustre/cmswork/dorigo/swgo/MT/Plots/Resolutions_R_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, epoch+startEpoch);
 #endif
 #ifdef UBUNTU
    sprintf (namepng,"/home/tommaso/Work/swgo/MT/Plots/Resolutions_R_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, epoch+startEpoch);
 #endif
 #ifdef INROOT
    sprintf (namepng,"./SWGO/Root/Resolutions_R_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, epoch+startEpoch);
 #endif
    Resor->Print(namepng);
    TCanvas * NER = new TCanvas ("NER", "",400,400);
    NER->cd();
    NvsER->Draw("COLZ");
    // Two scatterplots
    TCanvas * ResoSummary = new TCanvas ("ResoSummary", "", 1000, 400);
    ResoSummary->Divide(2,1);
    ResoSummary->cd(1);
    DE0vsER->Draw("COLZ");
    ResoSummary->cd(2);
    DR0vsER->Draw("COLZ");
 #ifdef STANDALONE
    sprintf (namepng,"/lustre/cmswork/dorigo/swgo/MT/Plots/Resolutions_ER_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, epoch+startEpoch);
 #endif
 #ifdef UBUNTU
    sprintf (namepng,"/home/tommaso/Work/swgo/MT/Plots/Resolutions_ER_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, epoch+startEpoch);
 #endif
 #ifdef INROOT
    sprintf (namepng,"./SWGO/Root/Resolutions_ER_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, epoch+startEpoch);
 #endif
#endif // PLOTRESOLUTIONS

    // Plot histos of residuals in X0, Y0
    // ----------------------------------
    /*TCanvas * C1 = new TCanvas ("C1","",1200,500);
    C1->Divide(5,2);
    C1->cd(1);
    DXP->Draw();
    C1->cd(2);
    DYP->Draw();
    C1->cd(3);
    DXG->Draw();
    C1->cd(4);
    DYG->Draw();
    C1->cd(5);
    DTHP->Draw();
    C1->cd(6);
    DPHP->Draw();
    C1->cd(7);
    DTHG->Draw();
    C1->cd(8);
    DPHG->Draw();
    C1->cd(9);
    DTHPvsT->Draw("COLZ");
    C1->cd(10);
    DTHGvsT->Draw("COLZ");
    */

#ifdef PLOTS
    // Plots of RMS vs R
    // -----------------
    for (int ib=0; ib<NbinsResR; ib++) {
        for (int jb=0; jb<NbinsResE; jb++) {
            double sumx2_XYg = 0.;
            double sumx2_T_g = 0.;
            double sumx2_P_g = 0.;
            double sumx2_E_g = 0.;
            double sumx2_XYp = 0.;
            double sumx2_T_p = 0.;
            double sumx2_P_p = 0.;
            double sumx2_E_p = 0.;
            double sumx_XYg = 0.;
            double sumx_T_g = 0.;
            double sumx_P_g = 0.;
            double sumx_E_g = 0.;
            double sumx_XYp = 0.;
            double sumx_T_p = 0.;
            double sumx_P_p = 0.;
            double sumx_E_p = 0.;
            double sumn_XYg = 0.;
            double sumn_T_g = 0.;
            double sumn_P_g = 0.;
            double sumn_E_g = 0.;
            double sumn_XYp = 0.;
            double sumn_T_p = 0.;
            double sumn_P_p = 0.;
            double sumn_E_p = 0.;
            int ihist = ib*NbinsResE+jb;
            for (int ibh=1; ibh<=100; ibh++) {
                double xy= maxdxy/100*(ibh-0.5);// maxdxy m in 100 bins
                double t = pi/2.*(ibh-0.5)/100; // pi in 100 bins
                double p = pi*(ibh-0.5)/100;    // pi/2 in 100 bins
                double e = 10.*(ibh-0.5)/100;       // 10 PeV in 100 bins
                double n = DXYg[ihist]->GetBinContent(ibh);
                sumn_XYg += n;
                sumx_XYg += n*xy;
                sumx2_XYg += n*xy*xy;
                n = DT_g[ihist]->GetBinContent(ibh);
                sumn_T_g += n;
                sumx_T_g += n*t;
                sumx2_T_g += n*t*t;
                n = DP_g[ihist]->GetBinContent(ibh);
                sumn_P_g += n;
                sumx_P_g += n*p;
                sumx2_P_g += n*p*p;
                n = DE_g[ihist]->GetBinContent(ibh);
                sumn_E_g += n;
                sumx_E_g += n*e;
                sumx2_E_g += n*e*e;
                n = DXYp[ihist]->GetBinContent(ibh);
                sumn_XYp += n;
                sumx_XYp += n*xy;
                sumx2_XYp += n*xy*xy;
                n = DT_p[ihist]->GetBinContent(ibh);
                sumn_T_p += n;
                sumx_T_p += n*t;
                sumx2_T_p += n*t*t;
                n = DP_p[ihist]->GetBinContent(ibh);
                sumn_P_p += n;
                sumx_P_p += n*p;
                sumx2_P_p += n*p*p;
                n = DE_p[ihist]->GetBinContent(ibh);
                sumn_E_p += n;
                sumx_E_p += n*e;
                sumx2_E_p += n*e*e;
            }
            if (sumn_XYg>0) sumx_XYg = sumx_XYg/sumn_XYg;
            if (sumn_T_g>0) sumx_T_g = sumx_T_g/sumn_T_g;
            if (sumn_P_g>0) sumx_P_g = sumx_P_g/sumn_P_g;
            if (sumn_E_g>0) sumx_E_g = sumx_E_g/sumn_E_g;
            if (sumn_XYp>0) sumx_XYp = sumx_XYp/sumn_XYp;
            if (sumn_T_p>0) sumx_T_p = sumx_T_p/sumn_T_p;
            if (sumn_P_p>0) sumx_P_p = sumx_P_p/sumn_P_p;
            if (sumn_E_p>0) sumx_E_p = sumx_E_p/sumn_E_p;
            if (sumn_XYg>0) sumx2_XYg = sumx2_XYg/sumn_XYg - pow(sumx_XYg,2.);
            if (sumn_T_g>0) sumx2_T_g = sumx2_T_g/sumn_T_g - pow(sumx_T_g,2.);
            if (sumn_P_g>0) sumx2_P_g = sumx2_P_g/sumn_P_g - pow(sumx_P_g,2.);
            if (sumn_E_g>0) sumx2_E_g = sumx2_E_g/sumn_E_g - pow(sumx_E_g,2.);
            if (sumn_XYp>0) sumx2_XYp = sumx2_XYp/sumn_XYp - pow(sumx_XYp,2.);
            if (sumn_T_p>0) sumx2_T_p = sumx2_T_p/sumn_T_p - pow(sumx_T_p,2.);
            if (sumn_P_p>0) sumx2_P_p = sumx2_P_p/sumn_P_p - pow(sumx_P_p,2.);
            if (sumn_E_p>0) sumx2_E_p = sumx2_E_p/sumn_E_p - pow(sumx_E_p,2.);
            XRMSvsRg->SetBinContent (ib+1,jb+1,sumx_XYg);
            TRMSvsRg->SetBinContent (ib+1,jb+1,sumx_T_g);
            PRMSvsRg->SetBinContent (ib+1,jb+1,sumx_P_g);
            ERMSvsRg->SetBinContent (ib+1,jb+1,sumx_E_g);
            XRMSvsRp->SetBinContent (ib+1,jb+1,sumx_XYp);
            TRMSvsRp->SetBinContent (ib+1,jb+1,sumx_T_p);
            PRMSvsRp->SetBinContent (ib+1,jb+1,sumx_P_p);
            ERMSvsRp->SetBinContent (ib+1,jb+1,sumx_E_p);
            
            // Rms: photons...
            // ---------------
            if (sumn_XYg>1. && sumx2_XYg>0.) {
                XRMSvsRg->SetBinError (ib+1,jb+1,sqrt(sumx2_XYg/(sumn_XYg-1)));
            } else if (sumx2_XYg>0.) {
                XRMSvsRg->SetBinError (ib+1,jb+1,sqrt(sumx2_XYg));
            }
            if (sumn_T_g>1 && sumx2_T_g>0.) {
                TRMSvsRg->SetBinError (ib+1,jb+1,sqrt(sumx2_T_g/(sumn_T_g-1)));
            } else if (sumx2_T_g>0.) {
                TRMSvsRg->SetBinError (ib+1,jb+1,sqrt(sumx2_T_g));
            }
            if (sumn_P_g>1 && sumx2_P_g>0.) {
                PRMSvsRg->SetBinError (ib+1,jb+1,sqrt(sumx2_P_g/(sumn_P_g-1)));
            } else if (sumx2_P_g>0.) {
                PRMSvsRg->SetBinError (ib+1,jb+1,sqrt(sumx2_P_g));
            }
            if (sumn_E_g>1 && sumx2_E_g>0.) {
                ERMSvsRg->SetBinError (ib+1,jb+1,sqrt(sumx2_E_g/(sumn_E_g-1)));
            } else if (sumx2_E_g>0.) {
                ERMSvsRg->SetBinError (ib+1,jb+1,sqrt(sumx2_E_g));
            }
            // Protons now
            if (sumn_XYp>1 && sumx2_XYp>0.) {
                XRMSvsRp->SetBinError (ib+1,jb+1,sqrt(sumx2_XYp/(sumn_XYp-1)));
            } else if (sumx2_XYp>0.) {
                XRMSvsRp->SetBinError (ib+1,jb+1,sqrt(sumx2_XYp));
            }
            if (sumn_T_p>1 && sumx2_T_p>0.) {
                TRMSvsRp->SetBinError (ib+1,jb+1,sqrt(sumx2_T_p/(sumn_T_p-1)));
            } else if (sumx2_T_p>0.) {
                TRMSvsRp->SetBinError (ib+1,jb+1,sqrt(sumx2_T_p));
            }
            if (sumn_P_p>1 && sumx2_P_p>0.) {
                PRMSvsRp->SetBinError (ib+1,jb+1,sqrt(sumx2_P_p/(sumn_P_p-1)));
            } else if (sumx2_P_p>0.) {
                PRMSvsRp->SetBinError (ib+1,jb+1,sqrt(sumx2_P_p));
            }
            if (sumn_E_p>1 && sumx2_E_p>0.) {
                ERMSvsRp->SetBinError (ib+1,jb+1,sqrt(sumx2_E_p/(sumn_E_p-1)));
            } else if (sumx2_E_p>0.) {
                ERMSvsRp->SetBinError (ib+1,jb+1,sqrt(sumx2_E_p));
            }
        } // End jb
    } // End ib
    
    // Ready to plot these
    // -------------------
    TCanvas * RMSvsRg = new TCanvas ("RMSvsRg","RMS of par estimates vs R for photons", 800, 800);
    RMSvsRg->Divide(2,2);
    RMSvsRg->cd(1);
    XRMSvsRg->Draw("COLZ");
    RMSvsRg->cd(2);
    TRMSvsRg->Draw("COLZ");
    RMSvsRg->cd(3);
    PRMSvsRg->Draw("COLZ");
    RMSvsRg->cd(4);
    ERMSvsRg->Draw("COLZ");

    TCanvas * RMSvsRp = new TCanvas ("RMSvsRp","RMS of par estimates vs R for protons", 800, 800);
    RMSvsRp->Divide(2,2);
    RMSvsRp->cd(1);
    XRMSvsRp->Draw("COLZ");
    RMSvsRp->cd(2);
    TRMSvsRp->Draw("COLZ");
    RMSvsRp->cd(3);
    PRMSvsRp->Draw("COLZ");
    RMSvsRp->cd(4);
    ERMSvsRp->Draw("COLZ");

    /*TCanvas * D_x = new TCanvas ("D_x", "", 800, 800);
    D_x->Divide(5,NbinsResR/5);
    for (int ib=0; ib<NbinsResR; ib++) {
        D_x->cd(ib+1);
        DXYg[ib]->SetLineColor(kBlue);
        DXYg[ib]->Draw();
        DXYp[ib]->SetLineColor(kRed);
        DXYp[ib]->Draw("SAME");
    }
    TCanvas * D_e = new TCanvas ("D_e", "", 800, 800);
    D_e->Divide(5,NbinsResR/5);
    for (int ib=0; ib<NbinsResR; ib++) {
        D_e->cd(ib+1);
        DE_g[ib]->SetLineColor(kBlue);
        DE_g[ib]->Draw();
        DE_p[ib]->SetLineColor(kRed);
        DE_p[ib]->Draw("SAME");
    }
    TCanvas * D_T = new TCanvas ("D_T", "", 800, 800);
    D_T->Divide(5,NbinsResR/5);
    for (int ib=0; ib<NbinsResR; ib++) {
        D_T->cd(ib+1);
        DT_g[ib]->SetLineColor(kBlue);
        DT_g[ib]->Draw();
        DT_p[ib]->SetLineColor(kRed);
        DT_p[ib]->Draw("SAME");
    }
    TCanvas * D_p = new TCanvas ("D_p", "", 800, 800);
    D_p->Divide(5,NbinsResR/5);
    for (int ib=0; ib<NbinsResR; ib++) {
        D_p->cd(ib+1);
        DP_g[ib]->SetLineColor(kBlue);
        DP_g[ib]->Draw();
        DP_p[ib]->SetLineColor(kRed);
        DP_p[ib]->Draw("SAME");
    }
    */

    TCanvas * C2 = new TCanvas ("C2","", 1200, 500);
    C2->Divide(3,2);
    C2->cd(1);
    LLRP->Draw();
    LLRG->Draw("SAME");
    C2->cd(2);
    LLRP->SetLineWidth(3);
    LLRP->Draw();
    LLRG->SetLineWidth(3);
    LLRP->SetLineColor(kRed);
    LLRG->Draw("SAME");
    C2->cd(3);
    SigLRT->Draw();
    C2->cd(4);
    SigLvsDRg->Draw("BOX");
    SigLvsDRp->SetLineColor(kRed);
    SigLvsDRp->Draw("SAMEBOX");
    C2->cd(5);
    NmuvsSh->Draw("BOX");
    C2->cd(6);
    NevsSh->Draw("BOX");
 
    // Plot of triggering probability vs distance and energy
    // -----------------------------------------------------
    for (int ib=1; ib<=25; ib++) {
        for (int jb=1; jb<=10; jb++) {
            double c = PAPvsD->GetBinContent(ib,jb);
            double n = NPAPvsD->GetBinContent(ib,jb);
            if (n>0) PPvsD->SetBinContent(ib,jb,c/n);
            c = PAGvsD->GetBinContent(ib,jb);
            n = NPAGvsD->GetBinContent(ib,jb);
            if (n>0) PGvsD->SetBinContent(ib,jb,c/n);
        }
    }
    TCanvas * Ptr = new TCanvas ("Ptr","",800,500);
    Ptr->Divide(1,2);
    Ptr->cd(1);
    PGvsD->Draw("COLZ");
    Ptr->cd(2);
    PPvsD->Draw("COLZ");
#endif

#ifdef PLOTS

    // Plot the distributions of fluxes per m^2
    // ----------------------------------------
    if (plotdistribs) {
        TH1D * MFG3 = new TH1D ("MFG3", "", 150, 0., 1500.);
        TH1D * EFG3 = new TH1D ("EFG3", "", 150, 0., 1500.);
        TH1D * MFP3 = new TH1D ("MFP3", "", 150, 0., 1500.);
        TH1D * EFP3 = new TH1D ("EFP3", "", 150, 0., 1500.);
        TH1D * MFG4 = new TH1D ("MFG4", "", 150, 0., 1500.);
        TH1D * EFG4 = new TH1D ("EFG4", "", 150, 0., 1500.);
        TH1D * MFP4 = new TH1D ("MFP4", "", 150, 0., 1500.);
        TH1D * EFP4 = new TH1D ("EFP4", "", 150, 0., 1500.);
        TH1D * MFG5 = new TH1D ("MFG5", "", 150, 0., 1500.);
        TH1D * EFG5 = new TH1D ("EFG5", "", 150, 0., 1500.);
        TH1D * MFP5 = new TH1D ("MFP5", "", 150, 0., 1500.);
        TH1D * EFP5 = new TH1D ("EFP5", "", 150, 0., 1500.);

        for (int i=0; i<150; i++) {
            double r = i*10.+0.5;
            MFG3->SetBinContent(i+1,MFromG(0.1,0,r,0)/TankArea);
            EFG3->SetBinContent(i+1,EFromG(0.1,0,r,0)/TankArea);
            MFP3->SetBinContent(i+1,MFromP(0.1,0,r,0)/TankArea);
            EFP3->SetBinContent(i+1,EFromP(0.1,0,r,0)/TankArea);
            MFG4->SetBinContent(i+1,MFromG(1.,0,r,0)/TankArea);
            EFG4->SetBinContent(i+1,EFromG(1.,0,r,0)/TankArea);
            MFP4->SetBinContent(i+1,MFromP(1.,0,r,0)/TankArea);
            EFP4->SetBinContent(i+1,EFromP(1.,0,r,0)/TankArea);
            MFG5->SetBinContent(i+1,MFromG(10.,0,r,0)/TankArea);
            EFG5->SetBinContent(i+1,EFromG(10.,0,r,0)/TankArea);
            MFP5->SetBinContent(i+1,MFromP(10.,0,r,0)/TankArea);
            EFP5->SetBinContent(i+1,EFromP(10.,0,r,0)/TankArea);
        }
        EFP3->SetMinimum(0.00000001);
        EFP3->SetMaximum(10000.);
        MFP3->SetMinimum(0.0000001);
        MFP3->SetMaximum(10.);
        EFG3->SetMinimum(0.00000001);
        EFG3->SetMaximum(10000.);
        MFG3->SetMinimum(0.00000001);
        MFG3->SetMaximum(1.);

        TCanvas * G = new TCanvas ("G","", 800, 800);
        G->Divide(2,2);
        G->cd(4);
        MFG3->Draw();
        MFG4->Draw("SAME");
        MFG5->Draw("SAME");
        G->cd(3);
        EFG3->Draw();
        EFG4->Draw("SAME");
        EFG5->Draw("SAME");
        G->cd(2);
        MFP3->Draw();
        MFP4->Draw("SAME");
        MFP5->Draw("SAME");
        G->cd(1);
        EFP3->Draw();
        EFP4->Draw("SAME");
        EFP5->Draw("SAME");
    }    
#endif

    // Write selected histograms to root file
    // --------------------------------------
    // The def below has been done earlier, keeping it commented here in case we wish to remove the block of RatioEG fluxes
//#ifdef STANDALONE
//    string rootPath = GlobalPath + "Root/"; // "/lustre/cmswork/dorigo/swgo/MT/Root/";
//#endif
//#ifdef INROOT
//    string rootPath = "./SWGO/Root/";
//#endif
    if (!scanU) { // otherwise CT is not defined
        std::stringstream rootstr;
        char rnum[120];
        sprintf (rnum,"Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile);
        rootstr << "Swgolo141";
        string namerootfile = rootPath  + rootstr.str() + rnum + ".root";
        TFile * rootfile = new TFile (namerootfile.c_str(),"RECREATE");
        rootfile->cd();
        CT->Write();

#ifdef PLOTRESOLUTIONS
        Resoe->Write();
        Resor->Write();
        ResoSummary->Write();
#endif
#ifdef PLOTS
        Ptr->Write();
        C1->Write();
        C2->Write();
        //C->Write();
/*    if (checkmodel) {
        TMPe0->Write();
        TMPe1->Write();
        TMPe2->Write();
        TMPe3->Write();
        TMPm0->Write();
        TMPm1->Write();
        TMPm2->Write();
        TMPm3->Write();
        mgflux->Write();
        mpflux->Write();
        egflux->Write();
        epflux->Write();
    } */
#endif
        rootfile->Close();
    }

    // End of program
    // --------------
    cout    << endl;
    cout    << "     The program terminated correctly. " << endl;
    cout    << "     Warnings: " << endl;
    cout    << "     1 - " << warnings1 << endl;
    cout    << "     2 - " << warnings2 << endl;
    cout    << "     3 - " << warnings3 << endl;
    cout    << "     4 - " << warnings4 << endl;
    cout    << "     5 - " << warnings5 << endl;
    cout    << "     6 - " << warnings6 << endl;
    cout    << "     7 - " << warnings7 << endl;
    cout    << "     *****************************************************************" << endl;
    cout    << endl;
    
    // Close dump file
    // ---------------
    outfile << endl;
    outfile << "     The program terminated correctly. " << endl;
    outfile << "     Warnings: " << endl;
    outfile << "     1 - " << warnings1 << endl;
    outfile << "     2 - " << warnings2 << endl;
    outfile << "     3 - " << warnings3 << endl;
    outfile << "     4 - " << warnings4 << endl;
    outfile << "     5 - " << warnings5 << endl;
    outfile << "     6 - " << warnings6 << endl;
    outfile << "     7 - " << warnings7 << endl;
    outfile << "--------------------------------------------------------------" << endl;
    outfile.close();

    // If requested, write output geometry to file
    // -------------------------------------------
    if (writeGeom) SaveLayout(); 

    return 0;
}

// Service routine, not integrated in swgolo()
// Function that studies the probability of a shower to pass a trigger threshold on the
// number of detectors seeing >0 particles
// ------------------------------------------------------------------------------------
void CheckProb (int Ntrigger=50, int Nev=1000, int N_trials=100, int Nu=90, int sh=3,
                double Spacing=50., double SpacingStep = 50., double Rsl=300.) {

    // Pass parameters:
    // ----------------
    // Nu               = Number of detector elements. For radial distr, use 1/7/19/37/61/91/127/169/217/271/331/397/469/547/631/721...
    // Spacing          = Initial spacing of tanks
    // SpacingStep      = Increase in spacing 
    // shape            = Geometry of the initial layout (0=hexagonal, 1=taxi, 2=spiral)

    // UNITS
    // -----
    // Position: meters
    // Angle:    radians
    // Time:     nanoseconds
    // Energy:   PeV

    if (Ntrigger>maxNtrigger) {
        cout << "Ntrigger cannot exceed " << maxNtrigger << endl;
        return;
    }

    // Get static values from pass parameters
    // --------------------------------------
    Nunits          = Nu;
    DetectorSpacing = Spacing;
    Rslack          = Rsl;
    shape           = sh;

    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);

    // Get a sound RN generator
    // ------------------------
    delete gRandom;
    myRNG = new TRandom3();
    myRNG2= new TRandom3();

    // Suppress root warnings
    gROOT->ProcessLine( "gErrorIgnoreLevel = 6001;");
    gROOT->ProcessLine( "gPrintViaErrorHandler = kTRUE;");
 
    // Define the current geometry 
    // ---------------------------
    if (readGeom) {
        ReadLayout ();
    } else {
        DefineLayout();
    }
    
    // Read in parametrizations of particle fluxes and lookup table
    // ------------------------------------------------------------
    int code = ReadShowers ();
    if (code!=0) {
        cout << "Trouble reading showers, terminating. " << endl;
        return;
    }

    // Set fluxes of backgrounds
    // -------------------------
    TankArea   = pow(TankRadius,2.)*pi*TankNumber;
    fluxB_mu   = TankArea*Bgr_mu_per_m2;
    fluxB_e    = TankArea*Bgr_e_per_m2;

    TH1D * DPp[4]; 
    TH2D * PfTvsPfCp[4];
    TH1D * DPg[4]; 
    TH2D * PfTvsPfCg[4];
    char namedp[30];
    for (int ie=0; ie<4; ie++) {
        sprintf (namedp, "DPg%d",ie);
        DPg[ie] = new TH1D(namedp,namedp, 100, -0.2, 0.2);
        sprintf (namedp, "PfTvsPfCg%d",ie);
        PfTvsPfCg[ie] = new TH2D(namedp,namedp, 26, 0., 1.04, 26, 0., 1.04);
        sprintf (namedp, "DPp%d",ie);
        DPp[ie] = new TH1D(namedp,namedp, 100, -0.2, 0.2);
        sprintf (namedp, "PfTvsPfCp%d",ie);
        PfTvsPfCp[ie] = new TH2D(namedp,namedp, 26, 0., 1.04, 26, 0., 1.04);
    }

    // Loop on g, p hypotheses
    // -----------------------
    bool Gamma;
    for (int type=0; type<2; type++) {
        if (type==0) Gamma = true;
        if (type==1) Gamma = false;

        // Loop on 4 energy points
        // -----------------------
        for (int ie=0; ie<4; ie++) {
            double E;
            if (type==0) {
                E = 0.2+pow(ie,2.)*0.3;
            } else {
                E = 0.1+pow(ie,2.)*0.1;
            }
            // Shower generation loop
            // ----------------------
            for (int is=0; is<Nev; is++) {
                if (is%100==0) cout << ie << " Event # " << is << endl;
                TrueE[is]     = E;
                TrueTheta[is] = 0.;
                TruePhi[is]   = 0.;
                TrueX0[is]    = myRNG->Uniform(-1000,1000);
                TrueY0[is]    = myRNG->Uniform(-1000,1000);

                double mug = 0.;
                double  eg = 0.;
                double mup = 0.;
                double  ep = 0.;
                // Compute a-priori probability that shower passes trigger
                // -------------------------------------------------------
                double ExpN_ge1 = 0.; // Expectation value of number of detectors with >=1 particle seen
                for (int id=0; id<Nunits; id++) {
                    double nptot = 0.; // Exp value of particles in detector
                    double R = EffectiveDistance(x[id],y[id],TrueX0[is],TrueY0[is],TrueTheta[is],TruePhi[is],0);
                    if (Gamma) {
                        mug = MFromG (E,0.,R,0) + fluxB_mu;
                        eg  = EFromG (E,0.,R,0) + fluxB_e;
                        nptot = mug+eg;
                    } else {
                        mup = MFromP (E,0.,R,0) + fluxB_mu;
                        ep  = EFromP (E,0.,R,0) + fluxB_e;
                        nptot = mup+ep; 
                    }
                    // Each unit in the id loop is a macrotank constituted by TankNumber tanks. The Ntrigger condition
                    // applies to the number of tanks so we need to account for this (Ncount). The calculation of 
                    // SumProbGe1 relies on the expected number of tanks with >=1 particle if the macrotank has Npexp
                    // -----------------------------------------------------------------------------------------------
                    ExpN_ge1 += TankNumber * (1. - exp(-nptot/TankNumber)); // 1.-exp(-Npexp);     
                }
                double P_avg = ExpN_ge1 / (Nunits*TankNumber); // Average detection probability for the Nunits*TankNumber tanks

                // Compute P(N>Ntrigger) from approx formula
                // -----------------------------------------
                double Approx_prob = 1.;
                for (int j=0; j<Ntrigger; j++) {
                        // Poisson approximation:
                        // ----------------------
                        Approx_prob -= exp(-ExpN_ge1)*pow(ExpN_ge1,j)/Factorial(j);
                }

                // Resample particles for shower
                // -----------------------------
                double ProbFromTrials = 0.;
                bool Units[maxTankNumber]; // This flag is set on is a unit in a macroaggregate has seen at least one particle
                for (int k=0; k<N_trials; k++) {
                    int Counts = 0;
                    for (int id=0; id<Nunits; id++) {
                        double R = EffectiveDistance (x[id],y[id],TrueX0[is],TrueY0[is],0.,0.,0);
                        int nm = 0;
                        int ne = 0;
                        if (Gamma) {
                            mug = MFromG(E,0.,R,0) + fluxB_mu;
                            eg  = EFromG(E,0.,R,0) + fluxB_e;
                            if (mug>0.) nm = myRNG->Poisson(mug); // Otherwise it remains zero
                            if (eg>0.)  ne = myRNG->Poisson(eg);  // Otherwise it remains zero
                        } else {
                            mup = MFromP(E,0.,R,0) + fluxB_mu;
                            ep  = EFromP(E,0.,R,0) + fluxB_e;
                            if (mup>0.) nm = myRNG->Poisson(mup); // Otherwise it remains zero
                            if (ep>0.)  ne = myRNG->Poisson(ep);  // Otherwise it remains zero
                        }
                        int Nptot = nm+ne;
                        // Now distribute at random Nptot into the TankNumber units
                        // --------------------------------------------------------
                        for (int i=0; i<TankNumber; i++) {Units[i]=false;};
                        for (int ip=0; ip<Nptot; ip++) {
                            int thistn = (int)(myRNG->Uniform(TankNumber));
                            if (Units[thistn]==false) {
                                Units[thistn] = true; 
                                Counts++;
                            }
                        }
                        // If we want to model
                        // a realization of a given number Nm+Ne particles distributed in TN tanks (for the purpose of
                        // checking if Ntrigger is satisfied with the generated shower, rather than computing the probability
                        // of triggering of the expected flux), we need to use Bernoulli trials, not the Poisson distribution.
                        // The probability that a tank in a macroaggregate of TN is not hit by a particle is (TN-1)/TN, so
                        // we have <N_empty> = TN*[(TN-1)/TN)]^(Nm+Ne), so N_signal = TN * {1-[(TN-1)/TN]^(Nm+Ne)}
                        // However, correlations screw up the above model, so better to do the direct calculation above.
                        // ---------------------------------------------------------------------------------------------------
                        //  Counts += TankNumber * (1.-pow(1.*(TankNumber-1)/TankNumber,nm+ne));
                    } // End loop on units
                    if (Counts>=Ntrigger) ProbFromTrials++;
                } // End loop on trials

                // Compute P(N>Ntrigger) from trials
                // ---------------------------------
                ProbFromTrials /= N_trials;
                //if (is%20==0) cout << ExpN_ge1 << " " << Approx_prob << " " << ProbFromTrials << endl;
                // cout << " " << ProbFromTrials << endl;

                if (type==0) {
                    PfTvsPfCg[ie]->Fill(ProbFromTrials,Approx_prob);
                    DPg[ie]->Fill(Approx_prob-ProbFromTrials);
                } else {
                    PfTvsPfCp[ie]->Fill(ProbFromTrials,Approx_prob);
                    DPp[ie]->Fill(Approx_prob-ProbFromTrials);
                }
            } // End is cycle
        } // End ie cycle
    } // End type cycle

    /*TCanvas * DPC = new TCanvas ("DPC", "", 700,700);
    DPC->Divide (4,2);
    for (int ie=0; ie<4; ie++) {
        DPC->cd(2*ie+1);
        DPC->GetPad(2*ie+1)->SetLogz();
        PfTvsPfCg[ie]->SetLineColor(kRed);
        PfTvsPfCp[ie]->SetLineColor(kBlue);
        PfTvsPfCg[ie]->Draw("BOX");
        DPC->cd(2*ie+2);
        DPC->GetPad(2*ie+2)->SetLogz();
        PfTvsPfCp[ie]->Draw("BOX");
    }
    */
    TCanvas * DPC2 = new TCanvas ("DPC", "", 700,700);
    DPC2->Divide (4,4);
    for (int ie=0; ie<4; ie++) {
        DPC2->cd(4*ie+1);
        DPC2->GetPad(4*ie+1)->SetLogz();
        PfTvsPfCg[ie]->SetLineWidth(3);
        PfTvsPfCg[ie]->SetLineColor(kRed);
        PfTvsPfCg[ie]->Draw("COLZ");
        DPC2->cd(4*ie+2);
        DPC2->GetPad(4*ie+2)->SetLogy();
        DPg[ie]->SetLineWidth(3);
        DPg[ie]->SetLineColor(kRed);
        DPg[ie]->Draw();
        DPC2->cd(4*ie+3);
        DPC2->GetPad(4*ie+3)->SetLogz();
        PfTvsPfCp[ie]->SetLineWidth(3);
        PfTvsPfCp[ie]->SetLineColor(kBlue);
        PfTvsPfCp[ie]->Draw("COLZ");
        DPC2->cd(4*ie+4);
        DPC2->GetPad(4*ie+4)->SetLogy();
        DPp[ie]->SetLineWidth(3);
        DPp[ie]->SetLineColor(kBlue);
        DPp[ie]->Draw();
    }
    return;
}

// Service routine, not integrated in swgolo()
// We model the interaction of a particle coming down with polar angle theta, from the positive x
// direction and at fixed y=y0, by assuming the detector upper side is at coordinate z0=0. We call x0
// the position of the particle at z0=0, with -R<x0<R+H/tan(theta). We proceed to find the 
// intersection of the path with the cylinder, and determine the total path length.
void CheckFlux (double H=3., double R=2.) {

    TH1D * PL[16]; 
    int nx = 100;
    int nt = 100;
    int ny = 100;
    TH2D * PLt[16];
    TH2D * PLx[16];
    char name[100];
    for (int i=0; i<16; i++) {
        sprintf (name,"PLt%d",i);
        PLt[i] = new TH2D (name, "Path length in tank vs impact parameter", nt, 0., 65., 100, 0., sqrt(H*H+4.*R*R)+1);
        sprintf (name,"PLx%d",i);
        PLx[i] = new TH2D (name, "Path length in tank vs distance", nx, -R, R+H*tan(thetamax), 100, 0., sqrt(H*H+4.*R*R)+1);
        sprintf (name,"PL%d",i);        
        PL[i]  = new TH1D (name, "Path length in tank", 100, 0., 5.);
    }
    TProfile2D * PLxy = new TProfile2D ("PLxy", "Path length vs xy on the ground", nx, -R, R+H*tan(thetamax), 32, -R, R, 0., 1000.);
    TProfile2D * PLzt = new TProfile2D ("PLzt", "Path length vs zt of particles", nt, 0., 65., 50, -H, 0., 0., 1000.);

    // We model the interaction of a particle coming down with polar angle theta, from the positive x
    // direction and at fixed y=y0, by assuming the detector upper side is at coordinate z0=0. We call x0
    // the position of the particle at z0=0, with -R<x0<R+H/tan(theta). We proceed to find the 
    // intersection of the path with the cylinder, and determine the total path length.
    // Cylinder: z = 0, z = -H; x^2+y^2 = R^2. Line: y=y0; z = (x-x0)/tan(theta)
    // x^2 = R^2 - y0^2  ->  (z*tan(theta) + x0)^2 = R^2 - y0^2
    // tan(theta)^2 * z^2 + 2 x0 tan(theta) * z + (x0^2 + y0^2 - R^2) = 0
    // z12 = [-2 x0 tan(theta) +- sqrt(4 x0^2 tan(th)^2 -4 tan(theta)^2(x0^2 + y0^2-R^2) ]/2 tan(theta)^2
    // --------------------------------------------------------------------------------------------------
    for (int ith=0; ith<nt; ith++) {
        double theta = (0.5+ith)/nt*thetamax; // Polar angle
        for (int id=0; id<ny; id++) { 
            int idy = 16*(id+0.5)/ny; // For plotting purposes
            double y0 = R*(id+0.5)/ny; // Impact parameter with center of tank
            for (int ix=0; ix<nx; ix++) {
                double x0 = -R + (0.5+ix)/nx*(2.*R+H*tan(theta)); // Intersection with upper plane
                double pathlength=0.;
                double z1 = 1.; // Just a random initial value
                double z2 = 1.;
                // Intersection with tank     
                // Upper point if hitting top of tank
                if (x0*x0<=R*R-y0*y0) {
                    z2 = 0.;
                }
                // Lower point if hitting bottom
                if (pow(-H*tan(theta)+x0,2.)<=R*R-y0*y0) {
                    z1 = -H;
                }
                double T = tan(theta);
                double T2 = T*T;
                double det = (R*R-y0*y0);
                if (z1==1.) z1 = (-x0-sqrt(det))/T;
                if (z2==1.) z2 = (-x0+sqrt(det))/T; 
//                    if (z1<-H) z1 = -H;
//                    if (z2>0.) z2 = 0.;
                    // Now compute path in tank
                if (z1>=-H && z2>=-H && z2<=0. && z1<=0.) { 
                    pathlength = (z2-z1)/cos(theta);
                    if (pathlength>5.) {
                        //cout << z1 << " " << z2 << " " << x0 << " " << y0 << " " << theta << " " << pathlength << endl; 
                    }
                    PLt[idy]->Fill(theta*180./pi,pathlength);
                    PLx[idy]->Fill(x0,pathlength);
                    PL[idy]->Fill(pathlength);
                    PLxy->Fill(x0,y0,pathlength);
                    PLxy->Fill(x0,-y0,pathlength);
                    PLzt->Fill(theta*180./pi,z2,pathlength);
                }
            }
        }
    }
    TCanvas * Ct = new TCanvas ("Ct","",800,800);
    Ct->Divide(4,4);
    for (int i=1; i<=16; i++) {
        Ct->cd(i);
        PLt[i-1]->Draw("COLZ");
    }
    TCanvas * Cx = new TCanvas ("Cx","",800,800);
    Cx->Divide(4,4);
    for (int i=1; i<=16; i++) {
        Cx->cd(i);
        PLx[i-1]->Draw("COLZ");
    }
    TCanvas * Cp = new TCanvas ("Cp","",800,800);
    Cp->Divide(4,4);
    for (int i=1; i<=16; i++) {
        Cp->cd(i);
        PL[i-1]->Draw("");
    }
    TCanvas * Cxy = new TCanvas ("Cxy","",1000,400);
    Cxy->cd();
    PLxy->Draw("COLZ");
    TCanvas * Czt = new TCanvas ("Czt","",1000,400);
    Czt->cd();
    PLzt->Draw("COLZ");

    return;
} 

// Service routine, not integrated in swgolo. Study flux in tanks
// We model the interaction of a particle coming down with polar angle theta, from the positive x
// direction and at fixed y=y0, by assuming the detector upper side is at coordinate z0=0. We call x0
// the position of the particle at z0=0, with -R<x0<R+H/tan(theta). We proceed to find the 
// intersection of the path with the cylinder, and determine the total path length.
// --------------------------------------------------------------------------------------------------
void CheckFlux2 (double H=3.6, double R=2.6) {

    int nx = 100;
    int nt = 100;
    int np = 100;
    int ny = 100;
    int nz = 100;
    double maxtheta = 65.*pi/180.;
    double rmax = sqrt(4*R*R+H*H);
    TProfile2D * PLxy = new TProfile2D ("PLxy", "Avg PL vs xy on the ground", 2*nx, -R-H*tan(maxtheta), R+H*tan(maxtheta), 2*ny, -R-H*tan(maxtheta), R+H*tan(maxtheta), 0., 1000.);
    TProfile2D * PLzt = new TProfile2D ("PLzt", "Avg PL vs height of side hit and theta of particles", nt, 0., 65., nz, -H, 0., 0., 1000.);
    TProfile2D * PLtd = new TProfile2D ("PLtd", "Avg Path length vs theta and distance of particles", nt, 0., 65., nx, 0., rmax, 0., 1000.);
    TProfile2D * PLxyt[20];
    char name[100];
    for (int i=0; i<20; i++) {
        sprintf (name,"PLxyt%d",i);
        PLxyt[i] = new TProfile2D (name, "Avg PL in tank vs theta", nx, -R-H*tan(maxtheta), R+H*tan(maxtheta), 
                                   ny, -R-H*tan(maxtheta), R+H*tan(maxtheta), 0., 1000.);
    }

    TH2D * F_tl = new TH2D ("F_tl","Flux (Theta,Path length)",  nt, 0.,65.,  100,0.,sqrt(4*R*R+H*H));
    TH2D * F_pl = new TH2D ("F_pl","Flux (Phirel,Path length)", np, 0.,180., 100,0.,sqrt(4*R*R+H*H));
    TH2D * F_rl = new TH2D ("F_rl","Flux (Radius,Path length)", nx, 0.,rmax, 100,0.,sqrt(4*R*R+H*H));

    // We model the interaction of a particle coming down with polar angle theta, from the positive x
    // direction and at fixed y=y0, by assuming the detector upper side is at coordinate z0=0. We call x0
    // the position of the particle at z0=0, with -R<x0<R+H/tan(theta). We proceed to find the 
    // intersection of the path with the cylinder, and determine the total path length.
    // Cylinder: z = 0, z = -H; x^2+y^2 = R^2. 
    // Line: y = y0 + (x-x0)tan(phi); z = -sqrt((x-x0)^2+(y-y0)^2)/tan(theta)
    // -------------------------------------------------------------------------------------------------- 
    for (int it=0; it<nt; it++) {
        double theta = (0.5+it)/nt*maxtheta; // polar angle
        double T = tan(theta);
        double T2 = T*T;
        for (int ip=0; ip<np; ip++) {
            double phi = (ip+0.5)/np*(2.*pi);
            for (int iy=0; iy<ny; iy++) { 
                double y0 = (iy+0.5)/ny*(R+H*tan(maxtheta)); 
                for (int ix=0; ix<nx; ix++) {
                    double x0 = (0.5+ix)/nx*(R+H*tan(maxtheta));         
                    double pathlength = 0.;
                    double phirel = phi-PhiFromXY(x0,y0);
                    if (phirel<0.) phirel+=twopi;
                    if (phirel>pi) phirel=pi-phirel;
                    double z1 = 1.; // Just a random initial value
                    double z2 = 1.;
                    // Intersection with tank     
                    // Upper point if hitting top of tank
                    if (x0*x0+y0*y0<=R*R) {
                        z2 = 0.;
                    }
                    // Lower point if hitting bottom
                    double sp = sin(phi);
                    double cp = cos(phi);
                    double tp = sp/cp;
                    double sp2 = sp*sp;
                    double cp2 = cp*cp;
                    // Line is z = (x-x0)/tan(theta)/cos(phi)
                    // or      z = (y-y0)/tan(theta)/sin(phi)
                    // so for z = -H we get the expressions below
                    double xmh = -H*T*cp + x0;
                    double ymh = -H*T*sp + y0;
                    if (xmh*xmh+ymh*ymh<=R*R) {
                        z1 = -H;
                    }
                    // Find intersections of line and cylinder
                    // We have 
                    //     x*x + y*y = R*R
                    // and
                    //     z = (x-x0)/T/cp  -> x = zTcp + x0
                    //     z = (y-y0)/T/sp  -> y = zTsp + y0
                    //     (zTcp+x0)^2 + (zTsp+y0)^2 -R*R = 0
                    //     z^2 T^2 + z (2Tcpx0+2Tspy0) + (x0^2+y0^2-R^2) = 0  
                    double a = T2;
                    double b = 2*T*cp*x0+2*T*sp*y0;
                    double c = x0*x0+y0*y0-R*R;
                    //double a = 1./cp2;
                    //double b = 2.*(tp-x0*tp);
                    //double c = y0*y0+pow(x0*tp,2.)-2.*x0*tp-R*R;
                    double discr = b*b-4.*a*c;
                    double z1side;
                    double z2side;
                    if (discr>=0.) {
                        z1side = (-b-sqrt(discr))/(2.*a);
                        z2side = (-b+sqrt(discr))/(2.*a);
//                        double x1side = (-b-sqrt(discr))/(2.*a);
//                        double x2side = (-b+sqrt(discr))/(2.*a);
//                        z1side = -sqrt(pow(x1side-x0,2.)/cp2)/T;
//                        z2side = -sqrt(pow(x2side-x0,2.)/cp2)/T;
                        if (z1side>z2side) {
                            double tmp = z1side;
                            z1side = z2side;
                            z2side = tmp;
                        }
                        if (z1==1.) z1 = z1side;
                        if (z2==1.) z2 = z2side;
                    }
                    //if (fabs(y0)>R) cout << "x0,y0 = " << x0 << "," << y0 << " th= " << theta << " ph= " << phi << " z1,z2 = " << z1 << " " << z2 << " " << endl;
                    
                    double r = sqrt(x0*x0+y0*y0);
                    if (z1>=-H && z2>=-H && z2<=0. && z1<=0.) { // this selects all and only tracks intersecting the tank
                        pathlength = (z2-z1)/cos(theta);

                        F_tl->Fill(theta*180./pi,pathlength);
                        F_pl->Fill(phirel*180./pi,pathlength);
                        F_rl->Fill(r,pathlength);
                    
                        // TProfiles
                        PLxy->Fill(x0,y0,pathlength);
                        PLxy->Fill(-y0,x0,pathlength);
                        PLxy->Fill(-x0,-y0,pathlength);
                        PLxy->Fill(y0,-x0,pathlength);
                        PLzt->Fill(theta*180./pi,z2,pathlength);
                        PLtd->Fill(theta*180./pi,r,pathlength);
                        int indth = it*20/nt;
                        PLxyt[indth]->Fill(x0,y0,pathlength);
                        PLxyt[indth]->Fill(-y0,x0,pathlength);
                        PLxyt[indth]->Fill(-x0,-y0,pathlength);
                        PLxyt[indth]->Fill(y0,-x0,pathlength);
                    } else { // All other tracks generated but missing the tank
                        pathlength = 0.;
                        //_tl->Fill(theta*180./pi,pathlength);
                        //F_pl->Fill(phirel*180./pi,pathlength);
                        //F_rl->Fill(r,pathlength);

                        // TProfiles
                        PLxy->Fill(x0,y0,pathlength);
                        PLxy->Fill(-y0,x0,pathlength);
                        PLxy->Fill(-x0,-y0,pathlength);
                        PLxy->Fill(y0,-x0,pathlength);
                        PLzt->Fill(theta*180./pi,z2,pathlength);
                        PLtd->Fill(theta*180./pi,r,pathlength);
                        int indth = it*20/nt;
                        PLxyt[indth]->Fill(x0,y0,pathlength);
                        PLxyt[indth]->Fill(-y0,x0,pathlength);
                        PLxyt[indth]->Fill(-x0,-y0,pathlength);
                        PLxyt[indth]->Fill(y0,-x0,pathlength);
                    }
                }
            }
        }
    }
    TCanvas * Cxy = new TCanvas ("Cxy","",1000,1000);
    Cxy->Divide(3,2);
    Cxy->cd(1);
    PLxy->Draw("COLZ");
    Cxy->cd(2);
    PLzt->Draw("COLZ");
    Cxy->cd(3);
    PLtd->Draw("COLZ");
    Cxy->cd(4);
    F_tl->Draw("COLZ");
    Cxy->cd(5);
    F_pl->Draw("COLZ");
    Cxy->cd(6);
    F_rl->Draw("COLZ");
    TCanvas * Cxyt = new TCanvas ("Cxyt","",1250,1000);
    Cxyt->Divide(5,4);
    for (int i=1; i<=20; i++) {
        Cxyt->cd(i);
        PLxyt[i-1]->Draw("COLZ");
    }

    return;
} 

// Service routine, not integrated in swgolo()
// Study the distribution of the average time recorded in a -IntegrationWindow/2,IntegrationWindow/2 range
// for N particles of which fb*N are coming from a Uniform distribution and (1-fb)*N from a Gaussian centered
// at time tg and width sigma_time
// ----------------------------------------------------------------------------------------------------------
std::pair<double,double> AverageTime (int N, double fb, double tg, int Ntr=10000, bool plot=false) {
    int Nbins = 100;
    TH1D * pdf = new TH1D ("pdf","",Nbins, -IntegrationWindow/2.,IntegrationWindow/2.);
    for (int i=0; i<Ntr; i++) {
        double t = 0;
        for (int j=0; j<N; j++) {
            double r = myRNG->Uniform();
            if (r<fb) {
                t += myRNG->Uniform(-IntegrationWindow/2.,IntegrationWindow/2.);
            } else {
                t += myRNG->Gaus(tg,sigma_time);
            }
        }
        t = t/N;
        pdf->Fill(t);
    }
    if (plot) {
        TCanvas * P = new TCanvas ("P", "", 500,500);
        P->cd();
        pdf->Draw();
    }

    // The approximate variance is a sum of the variance of the gaussian and the uniform. The 
    // linear (1-fb) and fb weights come from squares divided by linear factors at the denominators
    // -------------------------------------------------------------------------------------------- 
    double approx_sigma = sqrt((1-fb)*sigma_time*sigma_time/N + fb*IntegrationWindow*IntegrationWindow/(12.*N));
    TF1 * gpdf = new TF1 ("gpdf","[0]*exp(-0.5*pow((x-[1])/[2],2.))",-IntegrationWindow/2.,IntegrationWindow/2.);
    double p0 = 1./(sqrt2pi*approx_sigma)*Ntr*IntegrationWindow/Nbins;
    double p1 = tg*(1-fb); // The gaussian mean results from the fb fraction having 0 mean
    double p2 = approx_sigma;
    gpdf->SetParameters(p0,p1,p2);
    gpdf->SetLineColor(kRed);
    if (plot) gpdf->Draw("SAME");
    cout << "approx rms = " << approx_sigma << endl;
    double hrms = pdf->GetRMS();
    delete pdf;
    return std::make_pair(hrms,approx_sigma);
}

// Service routine, not integrated in swgolo()
// Use above routine AverageTime() to study the residuals of the sigma of the time distribution versus N and bgr fraction
// ----------------------------------------------------------------------------------------------------------------------
void ValidateApproxAverTime(int maxN=20, int maxF=10, int Ntr=1000) {
    TH2D * R = new TH2D ("R","", maxN, 0.5, maxN+0.5, maxF, 0., 1.);
    for (int in=0; in<maxN; in++) {
        for (int i_f=0; i_f<maxF; i_f++) {
            int N = in+1;
            double f = (i_f+0.5)/maxF;
            std::pair <double,double> p = AverageTime(N,f,0.,Ntr,false);
            double num = p.first;
            double den = p.second;
            if (den!=0.) R->SetBinContent(in+1, i_f+1, num/den);
        }
    }
    TCanvas * VAAT = new TCanvas ("VAAT","", 500,500);
    VAAT->cd();
    R->Draw("COL4");
    return;
}

// Service routine, not integrated in swgolo()
// -------------------------------------------
void Flux (double energy, double theta) {

    // Read in parametrizations of particle fluxes and lookup table
    // ------------------------------------------------------------
    int code = ReadShowers ();
    if (code!=0) {
        cout    << "     Unsuccessful retrieval of shower parameters from ascii files, terminating." << endl;
        return;
    }
    return;
}


