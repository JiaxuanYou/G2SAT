#include <getopt.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <assert.h>
#include <string>
using namespace std;

char *filename;
int maxxmin = 10;


void printUsage(){
  cout << "scalefree: analyzer of the scale-free structure of SAT instances. Version 2.3" << endl;
  cout << "Copyright (C) 2016  C. AnsÃ³tegui, M.L. Bonet, J. GirÃ¡ldez-Cru, J. Levy" << endl;
  cout << "Cite: C. AnsÃ³tegui, M. L. Bonet, J. Levy: \"On the Structure of Industrial SAT Instances\"." <<endl;
  cout << "Proc. of 15th Int. Conf. on Principles and Practice of Constraint Programming, CP 2009. Pages 127-141." << endl << endl;
  cout << "Usage: scalefree [options] <filename.cnf>" << endl;
  cout << "OPTIONS:" << endl;
  cout << "  -x <int>     : Max number of values discarded to estimate alpha or beta (default: 10)." << endl;
  exit(-1);
}

void parseArgs(int argc, char *argv[]){
  int opt;
  while((opt=getopt(argc, argv, "h?x:")) != -1)
    switch(opt){
    case 'x':
      maxxmin = atoi(optarg);
      break;
    case 'h':
    case '?':
    default:
      printUsage();
    }
  if(optind < argc)
    filename = argv[optind];
  else
    printUsage();
}


//-----------------------------------------------------------------------------
double powlawc(int x, int xmin, double alpha) {
//-----------------------------------------------------------------------------
// Computes sum_{i=x}^{\infty} x^{alpha} / sum_{i=xmin}^{\infty} x^{alpha}
// or approximates it as (x/xmin)^(alpha+1)
//-----------------------------------------------------------------------------
  assert(alpha < -1);
  assert(xmin <= x);
#define MAXITER 10000
  double num = 0, den = 0;
  int i;

  if (xmin < 25) {

    for (i=xmin; i<x; i++)
      den += pow(i, alpha);

    double pold = -2, p = -1;
    int n = 0;

    while (abs(p - pold) > 0.00000001 && n < MAXITER) {
      den += pow((double)i, alpha);
      num += pow((double)i, alpha);
      i++;
      n++;
      pold = p;
      p = num/den;
    }
    if (n < MAXITER)
      return p;
  }
  return pow((double)x/xmin, alpha + 1);
}

//-----------------------------------------------------------------------------
double exponc(int x, int xmin, double beta) {
//-----------------------------------------------------------------------------
  return exp(beta*(xmin - x)) ;
}


//-----------------------------------------------------------------------------
void arity(vector<pair <int,int> > &arityVar, vector<pair <int,int> > &arityCla){
//-----------------------------------------------------------------------------
  FILE *source;

  source = fopen(filename, "r");
  if(!source){
    cerr << "ERROR: Unable to read CNF file " << filename << endl;
    exit(-1);
  }

  // Skip comments
  int aux=-1;
  while((aux=getc(source))=='c')
    while (getc(source)!='\n');
  ungetc(aux,source);

  // File Head
  int totVars=0, totClauses=0;
  if( !fscanf(source, "p cnf %i %i", &totVars, &totClauses)) {
    cerr << "Invalid CNF file\n";
    exit(-1);
  }

  vector<int> nOccurs(totVars,0);  //nOccurs[i] = number of occurences of variable i+1
  vector<int> nSizes;              //nSizes[i] = number of clauses of size i

  int var;
  int size = 0;
  while(fscanf(source, "%d", &var)==1) {
    if(var == 0){
      if(size >= nSizes.size())
	nSizes.resize(size+1);
      nSizes[size]++;
      size=0;
    } else {
      size++;
      nOccurs[abs(var)-1]++;
    }
  }

  sort(nOccurs.begin(), nOccurs.end());

  int prev = nOccurs[0];
  int addition = 1;
  for (int i=1; i<nOccurs.size(); i++) {
    if (nOccurs[i] == prev)
      addition++;
    else {
      arityVar.push_back(make_pair(prev,addition));
      prev = nOccurs[i];
      addition = 1;
    }
  }
  arityVar.push_back(make_pair(prev,addition));

  for(int i=1; i<nSizes.size(); i++)
    if(nSizes[i]>0)
      arityCla.push_back(make_pair(i,nSizes[i]));
}


//-----------------------------------------------------------------------------
void mostlikely(vector <pair <int,int> > &v, string filename) {
//-----------------------------------------------------------------------------

  if (v.size() < 3) {
    cerr << "ERROR: Unable to compute exponent for "<<v.size()<<" points"<<endl;
    return;
  }

//---- Compute vectors x, y, sxy, sylogx ------------------

  int n=v.size();
  vector <double> x(n), y(n+1), syx(n+1), sylogx(n+1);

  double Sy = 0; // the sum of the occurences of all degrees
  for (int i=0; i<n; i++) Sy += v[i].second;

  syx[n] = 0;
  sylogx[n] = 0;
  y[n] = 0;
  for(int i=n-1; i>=0; i--) {
    x[i]      = v[i].first;
    y[i]      = y[i+1] + v[i].second / Sy; // Cumulative F(real)
    sylogx[i] = sylogx[i+1] + v[i].second / Sy * log(x[i]);
    syx[i]    = syx[i+1] + v[i].second / Sy *     x[i];
  }

  //------ Compute, for powerlaw (a) and exponential (b),
  //       the best alpha, xmin, dif and where is located--------------------

  double bestalpha, bestbeta;
  int    bestxmina=0, bestxminb=0;
  double bestdifa = 1, bestdifb = 1;
  int    bestinda, bestindb;
  int    wherea, whereb;

  int ind = 0;
  int xmin;


  for (int ind=0; ind<=maxxmin && ind<n-2; ind++) {
    xmin = (int)x[ind];



    double alpha = -1 - 1 / (sylogx[ind] / y[ind] - log((double)xmin - 0.5));


    double beta = log(1 / (syx[ind] / y[ind] - xmin) + 1);

    //------------- Model powerlaw -----------------------------------------

    double worstdif = -1;
    int worstx = -1;

    for (int j=ind+1; j<n; j++) {
      double aux;
      aux = abs((double)y[j]/y[ind] - powlawc((int)x[j],xmin,alpha));
      if (aux >= bestdifa) {
	worstdif = aux;
	worstx = (int)x[j];
	j = n;  //Finish search of worst diff
      } else if (aux >= worstdif) {
	worstdif = aux;
	worstx = (int)x[j];
      }
    }
    for (int j=ind; j<n; j++) {
      if (x[j] + 1 < x[j+1]) {
	double aux;
	aux = abs((double)y[j+1]/y[ind] - powlawc((int)x[j]+1,xmin,alpha));
	if (aux >= bestdifa) {
	  worstdif = aux;
	  worstx = (int)x[j]+1;
	  j = n;  //Finish search of worst diff
	} else if (aux >= worstdif) {
	  worstdif = aux;
	  worstx = (int)x[j]+1;
	}
      }
    }
    if(worstdif < bestdifa) {
      bestalpha = alpha;
      bestxmina = xmin;
      bestdifa = worstdif;
      bestinda = ind;
      wherea = worstx;
    }

    //------------- Model exponential -----------------------------------------
    worstdif = -1;
    worstx = -1;

    for (int j=ind+1; j<n; j++) {
      double aux;
      aux = abs((double)y[j]/y[ind] - exponc(x[j],xmin,beta));
      if (aux >= bestdifb) {
	worstdif = aux;
	worstx = x[j];
	j = n;  //Finish search of worst diff
      } else if (aux >= worstdif) {
	worstdif = aux;
	worstx = x[j];
      }
    }
    for (int j=ind; j<n; j++) {
      if (x[j] + 1 < x[j+1]) {
	double aux;
	aux = abs((double)y[j+1]/y[ind] - exponc(x[j]+1,xmin,beta));
	if (aux >= bestdifb) {
	  worstdif = aux;
	  worstx = x[j]+1;
	  j = n;  //Finish search of worst diff
	} else if (aux >= worstdif) {
	  worstdif = aux;
	  worstx = x[j]+1;
	}
      }
    }
    if(worstdif < bestdifb) {
      bestbeta = beta;
      bestxminb = xmin;
      bestdifb = worstdif;
      bestindb = ind;
      whereb = worstx;
    }
  }

  //--------- Write results --------------------------------------------

  cout << "  Powerlaw:\n";
  cout << "    alpha = " << -bestalpha << endl;
  cout << "    min. value = " << bestxmina << endl;
  cout << "    max. error = " << bestdifa << endl; //" in value:"<< wherea << endl;
  cout << "  Exponential:\n";
  cout << "    beta = " << bestbeta << endl;
  cout << "    min. value = " << bestxminb << endl;
  cout << "    max. error = " << bestdifb << endl; //" in value:"<< whereb << endl;
  //
  // //--------- Generate file INT-----------------------------------------
  //
  // string fileint(filename);
  // fileint.append(".int");
  // FILE *fint = fopen(fileint.c_str(), "w");
  // if(fint == NULL) {
  //   cerr << "ERROR: Unable to open file " << fileint << endl;
  //   exit(-1);
  // }
  //
  // for (int i=0; i<n; i++)
  //   fprintf(fint,"%d %f\n", (int)x[i], y[i]);
  // fclose(fint);
  //
  //
  // //-----------Generate file PLT------------------------------------------------
  //
  // string fileplt(filename);
  // fileplt.append(".plt");
  // FILE *fplot = fopen(fileplt.c_str(), "w");
  // if(fplot == NULL) {
  //   cerr << "ERROR: Unable to open file " << fileplt << endl;
  //  exit(-1);
  // }
  //
  // string eps(filename);
  // eps.append(".eps");
  //
  // fprintf(fplot,"set logscale xy\nset term postscript eps enhanced color\nset size 0.7,0.7\n");
  // fprintf(fplot,"set yrange [%.10f:1.1]\n", pow(10,(int)(log(y[n-1])/log(10))-1));
  // fprintf(fplot,"set output \"%s\"\n", eps.c_str());
  // fprintf(fplot, "plot \"%s\" ti \"variables\" lt 1 pt 7",fileint.c_str());
  // fprintf(fplot, ",%lf * x**%lf lt 1 ti \"{/Symbol a}=%0.2f\"", (double)y[bestinda]/ pow(bestxmina,bestalpha+1), bestalpha+1, -bestalpha);
  // fprintf(fplot, ",%lf*exp(-%lf*(x - %d)) lt 2 ti \"{/Symbol b}=%0.3f\"",  (double)y[bestindb], bestbeta, bestxminb, bestbeta);
  // fprintf(fplot, ",\"< echo '%f %f'\" ti \"\"\n", (double)x[bestinda], y[bestinda]);
  // fprintf(fplot, "quit\n");
  //
  // fclose(fplot);
  //
  // //------------- Try to gnuplot PLT file ------------------------------------
  //
  // string com("gnuplot ");
  // com.append(fileplt);
  // system(com.c_str());
}



//-----------------------------------------------------------------------------
int main(int argc, char *argv[]){
//-----------------------------------------------------------------------------
  vector <pair <int,int> > arityVar;
  vector <pair <int,int> > arityCla;
  double bestAlphaVar, bestBetaVar, bestAlphaCla, bestBetaCla;

  parseArgs(argc,argv);
  arity(arityVar, arityCla);

  string nameroot(filename);
  nameroot=nameroot.substr(0,nameroot.length()-4);

  string namevar(nameroot);
  namevar.append(".var");
  cout << "VARIABLES" << endl;
  mostlikely(arityVar, namevar);

  if (arityCla.size() > 2) {
    string namecla(nameroot);
    namecla.append(".cla");
    cout << "CLAUSES" << endl;
    mostlikely(arityCla, namecla);
  }
}
