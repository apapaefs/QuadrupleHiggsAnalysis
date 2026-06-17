/* TopHist 1.51 (310811) */
/* 1.3 (020610) */
/* 1.31 (140910) */
/* 1.32 (280910): - SET INTENSITY 4 default setting */
/* 1.4 (280311): class member variables now public; added functions add, subtract and divide to manipulate histograms more easily. Use with caution. Known bug: the cout number of entries is 0 and falling into histo is nan for these histos*/
/* 1.5 (010611) Added the TopDist class to make y,x distributions (using tdfill) */
/* 1.51 (310811) Added sqr() function to forget about pow(x,2); */

#include <cstdlib> 
#include <ctime> 
#include <iostream>
#include <string>  
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace std;



class TopHist {
 public:
  TopHist() = default; //constructor
  TopHist(int bins, string output, string title, double xmin, double xmax);
  void plota(int bins, string output, string title, double xmin, double xmax);
  void plot(bool norm, bool log);
  void plot(bool norm, bool log, double normfactor);
  void plotd(bool norm, bool log, double normfactor);
  void plotds(bool norm, bool log, double normfactor);

  void add(string output, bool norm, bool log);
  void add(string output, bool norm, bool log, double normfactor);
  void addds(string output, bool norm, bool log, double normfactor);

  void addd(string output, bool norm, bool log, double normfactor);
  void thfill(double value);
  void thfill(double value, double weight);
  double getxmax();
  void subtract(TopHist one, TopHist two);
  void add(TopHist one, TopHist two);
  void divide(TopHist one, TopHist two);
  //private: 
  double entries[10000];
  double totalentries;
  int size;
  string output_t;
  string title_t;
  double xmin_t;
  double xmax_t;
  int bins_t;
  double interval_t; 
};

class TopDist {
 public:
  TopDist(); //constructor
  TopDist(int bins, string output, string title, double xmin, double xmax);
  void plot(bool norm, bool log);
  void add(string output, bool log);
  void tdfill(double yvalue, double xvalue);

  //private: 
  double entries[10000];
  double totalentries;
  double tot_entries_bin[10000];
  int size;
  string output_t;
  string title_t;
  double xmin_t;
  double xmax_t;
  int bins_t;
  double interval_t; 
};




class GnuHist2D {
 public:
  GnuHist2D(); //constructor
  GnuHist2D(int xbins, int ybins, double xmin, double xmax, double ymin, double ymax);
  GnuHist2D(string output, int xbins, int ybins, double xmin, double xmax, double ymin, double ymax);
  
  void plot(bool norm);
  void add(string output, bool norm);
  
  void ghfill(double xvalue, double yvalue);
 private: 
  double entries2[100][100];
  int size;
  string output_t;
  double xmin_t;
  double xmax_t;
  double ymin_t;
  double ymax_t;
  int xbins_t;
  int ybins_t;
  double yinterval_t;
  double xinterval_t;
};



TopDist::TopDist(int bins, string output, string title, double xmin, double xmax): output_t(output), title_t(title), xmin_t(xmin), xmax_t(xmax), bins_t(bins) {
  interval_t = (xmax_t - xmin_t) / bins_t;
  for(int ddd = 0; ddd < bins_t; ddd++) { entries[ddd] = 0; tot_entries_bin[ddd] = 0;}
  totalentries = 0;
  interval_t = (xmax - xmin) / bins_t;
}

void TopDist::tdfill(double value, double xvalue) { 
  totalentries++;
  if(xvalue < xmax_t && xvalue > xmin_t) {       
    // cout << "value " << value << endl;
    for(int ii = 0; ii < bins_t; ii++) { 
      if(xvalue <= ((ii+1) * interval_t + xmin_t) && xvalue > (ii * interval_t + xmin_t) ) {
	//	cout << value << " title = " << title_t << endl;
	entries[ii]+=value;
	tot_entries_bin[ii]++;
	//cout << "Adding to " << ii * interval_t + xmin_t << endl;
      }
    }
  }
}

void GnuHist2D::ghfill(double xvalue, double yvalue) { 
   if(xvalue < xmax_t && xvalue > xmin_t && yvalue > ymin_t && yvalue < ymax_t) {       
    for(int ii = 0; ii < xbins_t; ii++) { 
      for(int jj = 0; jj < ybins_t; jj++) {
	if(xvalue <= ((ii+1) * xinterval_t + xmin_t) && xvalue > (ii * xinterval_t + xmin_t) ) {
	  if(yvalue <= ((jj+1) * yinterval_t + ymin_t) && yvalue > (jj * yinterval_t + ymin_t) ) {
	    //    cout << "xvalue = " << xvalue << " yvalue = " << yvalue << " falls into ii,jj: " << ii << ", " << jj << endl;
	    entries2[ii][jj]++;
	  }
	}
      }
    }
  }
  
}

GnuHist2D::GnuHist2D(int xbins, int ybins, double xmin, double xmax, double ymin, double ymax): xmin_t(xmin), xmax_t(xmax), ymax_t(ymax), ymin_t(ymin), xbins_t(xbins), ybins_t(ybins) {
  xinterval_t = (xmax_t - xmin_t) / xbins_t;
  yinterval_t = (ymax_t - ymin_t) / ybins_t;
  output_t = "gnu2d.dat";
  for(int ddd = 0; ddd < xbins_t; ddd++) { 
    for(int eee = 0; eee < ybins_t; eee++) { 
      entries2[ddd][eee] = 0;
    }
  }
 
}

GnuHist2D::GnuHist2D(string output, int xbins, int ybins, double xmin, double xmax, double ymin, double ymax): output_t(output), xmin_t(xmin), xmax_t(xmax), ymax_t(ymax), ymin_t(ymin), xbins_t(xbins), ybins_t(ybins) {
  xinterval_t = (xmax_t - xmin_t) / xbins_t;
  yinterval_t = (ymax_t - ymin_t) / ybins_t;
  for(int ddd = 0; ddd < xbins_t; ddd++) { 
    for(int eee = 0; eee < ybins_t; eee++) { 
      entries2[ddd][eee] = 0;
    }
  }
 
}


void GnuHist2D::plot(bool norm) {

  double totarea = 0;
  string titlecase = "";
  string left = "";
  string leftcase = "";
  // double plotarray[1000][1000];
  for(int ii = 0; ii < xbins_t; ii++) { 
    for(int jj = 0; jj < ybins_t; jj++) { 
      totarea += entries2[ii][jj];
    }
  }
  cout << output_t << " (2d), total entries = " << totarea << endl;
  
  if(!norm) { totarea = 1; }
  for(int kk = 0; kk < xbins_t; kk++) { 
    for(int ll = 0; ll < ybins_t; ll++) { 
      //cout << entries2[kk][ll] << endl;
      entries2[kk][ll] = entries2[kk][ll]/(totarea);
    }
  }

  ofstream outf;
  outf.open(output_t.c_str());
  for(int ii = 0; ii < xbins_t; ii++) { 
    for(int jj = 0; jj < ybins_t; jj++) {   
      //cout << (ii+1) * xinterval_t + xmin_t << "\t" <<  (jj+1) * yinterval_t + ymin_t << "\t"  << entries2[ii][jj] << endl;
      outf << (ii+1) * xinterval_t + xmin_t << "\t" <<  (jj+1) * yinterval_t + ymin_t << "\t"  << entries2[ii][jj] << endl;
    }
    outf << endl;

  }
  outf.close(); 
}



TopHist::TopHist(int bins, string output, string title, double xmin, double xmax): output_t(output), title_t(title), xmin_t(xmin), xmax_t(xmax), bins_t(bins) {
  interval_t = (xmax_t - xmin_t) / bins_t;
  for(int ddd = 0; ddd < bins_t; ddd++) { entries[ddd] = 0; }
  totalentries = 0;
  interval_t = (xmax - xmin) / bins_t;
}

void TopHist::plota(int bins, string output, string title, double xmin, double xmax) {
  double totnum = 0;
  string titlecase = "";
  string left = "";
  string leftcase = "";
  for(int jj = 0; jj < bins; jj++) { 
    totnum += entries[jj];
  }
  for(int kk = 0; kk < bins; kk++) { 
    entries[kk] = entries[kk]/totnum;
  }
  ofstream outf;
  outf.open(output.c_str()/*, fstream::app*/);
  outf << "NEW FRAME" << endl;
  outf << "SET WINDOW X 1.6 8 Y 1.6 9" << endl;
  outf << "SET FONT DUPLEX" << endl;
  outf << "TITLE TOP \""    << title     << "\"\n";
  outf << "CASE      \""    << titlecase << "\"\n";
  outf << "TITLE LEFT \""   << left      << "\"\n";
  outf << "CASE       \""   << leftcase  << "\"\n";
  outf << "SET INTENSITY 4\n";
  outf << "SET ORDER X Y DX\n";
  outf << "SET SCALE Y LOG\n";
  outf << "SET LIMITS X " << xmin << " " << xmax << endl;
  // cout << interval << endl;
  for(int ii = 0; ii < bins; ii++) { 
    outf << ii * interval_t + xmin + interval_t * 0.5 << "\t" << entries[ii] << endl;
  }
  outf << "HIST  " << endl;
  outf.close();
}

void TopDist::plot(bool norm, bool log) {
  double totnum = 0;
  string titlecase = "";
  string left = "";
  string leftcase = "";
  double *plotarray = new double[bins_t];

  //  cout << "total entries = " << totalentries << " falling into histo: " << totnum << endl;
  
  if(!norm) { totalentries = 1; }
  for(int kk = 0; kk < bins_t; kk++) { 
    plotarray[kk] = entries[kk]/tot_entries_bin[kk];
  }
  ofstream outf;
  outf.open(output_t.c_str());
  outf << "NEW FRAME" << endl;
  outf << "SET WINDOW X 1.6 8 Y 1.6 9" << endl;
  outf << "SET FONT DUPLEX" << endl;
  outf << "TITLE TOP \""    << title_t     << "\"\n";
  outf << "CASE      \""    << titlecase << "\"\n";
  outf << "TITLE LEFT \""   << left      << "\"\n";
  outf << "CASE       \""   << leftcase  << "\"\n";
  outf << "SET INTENSITY 4\n";
  outf << "SET ORDER X Y \n";
  if(log) { outf << "SET SCALE Y LOG\n"; }
  outf << "SET LIMITS X " << xmin_t << " " << xmax_t << endl;
  // cout << interval << endl;
  for(int ii = 0; ii < bins_t; ii++) { 
    outf << ii * interval_t + xmin_t + interval_t * 0.5 << "\t" << plotarray[ii] << endl;
  }
  outf << "HIST  " << endl;
  outf.close();
}

void TopHist::plot(bool norm, bool log) {
  double totnum = 0;
  string titlecase = "";
  string left = "";
  string leftcase = "";
  double *plotarray = new double[bins_t];
  for(int jj = 0; jj < bins_t; jj++) { 
    totnum += entries[jj];
  }  
  cout << title_t << ", N(attempted) = " << totalentries << ", N(in histo) = " << totnum << endl;
  if(!norm) { totnum = 1; }
  for(int kk = 0; kk < bins_t; kk++) { 
    plotarray[kk] = entries[kk]/totnum;
  }
  ofstream outf;
  outf.open(output_t.c_str());
  outf << "NEW FRAME" << endl;
  outf << "SET WINDOW X 1.6 8 Y 1.6 9" << endl;
  outf << "SET FONT DUPLEX" << endl;
  outf << "TITLE TOP \""    << title_t     << "\"\n";
  outf << "CASE      \""    << titlecase << "\"\n";
  outf << "TITLE LEFT \""   << left      << "\"\n";
  outf << "CASE       \""   << leftcase  << "\"\n";
  outf << "SET INTENSITY 4\n";
  outf << "SET ORDER X Y DX\n";
  if(log) { outf << "SET SCALE Y LOG\n"; }
  outf << "SET LIMITS X " << xmin_t << " " << xmax_t << endl;
  // cout << interval << endl;
  for(int ii = 0; ii < bins_t; ii++) { 
    outf << ii * interval_t + xmin_t + interval_t * 0.5 << "\t" << plotarray[ii] << endl;
  }
  outf << "HIST  " << endl;
  outf.close();
}

void TopHist::plot(bool norm, bool log, double normfactor) {
  double totnum = 0;
  string titlecase = "";
  string left = "";
  string leftcase = "";
  double *plotarray = new double[bins_t];
  for(int jj = 0; jj < bins_t; jj++) { 
    totnum += entries[jj];
  }  
  cout << title_t << ", N(attempted) = " << totalentries << ", N(in histo) = " << totnum << endl;
  if(!norm) { totnum = 1; }
  for(int kk = 0; kk < bins_t; kk++) { 
    plotarray[kk] = normfactor * entries[kk]/totnum;
  }
  ofstream outf;
  outf.open(output_t.c_str());
  outf << "NEW FRAME" << endl;
  outf << "SET WINDOW X 1.6 8 Y 1.6 9" << endl;
  outf << "SET FONT DUPLEX" << endl;
  outf << "TITLE TOP \""    << title_t     << "\"\n";
  outf << "CASE      \""    << titlecase << "\"\n";
  outf << "TITLE LEFT \""   << left      << "\"\n";
  outf << "CASE       \""   << leftcase  << "\"\n";
  outf << "SET INTENSITY 4\n";
  outf << "SET ORDER X Y DX\n";
  if(log) { outf << "SET SCALE Y LOG\n"; }
  outf << "SET LIMITS X " << xmin_t << " " << xmax_t << endl;
  // cout << interval << endl;
  for(int ii = 0; ii < bins_t; ii++) { 
    outf << ii * interval_t + xmin_t + interval_t * 0.5 << "\t" << plotarray[ii] << endl;
  }
  outf << "HIST  " << endl;
  outf.close();
}

void TopHist::plotd(bool norm, bool log, double normfactor) {
  double totnum = 0;
  string titlecase = "";
  string left = "";
  string leftcase = "";
  double *plotarray = new double[bins_t];
  for(int jj = 0; jj < bins_t; jj++) { 
    totnum += entries[jj];
  }
  if(!norm) { totnum = 1; }
  for(int kk = 0; kk < bins_t; kk++) { 
    plotarray[kk] = normfactor * bins_t * entries[kk]/ (totnum * (xmax_t - xmin_t) );
  }
  ofstream outf;
  outf.open(output_t.c_str());
  outf << "NEW FRAME" << endl;
  outf << "SET WINDOW X 1.6 8 Y 1.6 9" << endl;
  outf << "SET FONT DUPLEX" << endl;
  outf << "TITLE TOP \""    << title_t     << "\"\n";
  outf << "CASE      \""    << titlecase << "\"\n";
  outf << "TITLE LEFT \""   << left      << "\"\n";
  outf << "CASE       \""   << leftcase  << "\"\n";
  outf << "SET INTENSITY 4\n";
  outf << "SET ORDER X Y DX\n";
  if(log) { outf << "SET SCALE Y LOG\n"; }
  outf << "SET LIMITS X " << xmin_t << " " << xmax_t << endl;
  // cout << interval << endl;
  for(int ii = 0; ii < bins_t; ii++) { 
    outf << ii * interval_t + xmin_t + interval_t * 0.5 << "\t" << plotarray[ii] << endl;
  }
  outf << "HIST  " << endl;
  outf.close();
}




void TopHist::thfill(double value) { 
  totalentries++;
  if(value < xmax_t && value > xmin_t) {       
    // cout << "value " << value << endl;
    for(int ii = 0; ii < bins_t; ii++) { 
      if(value <= ((ii+1) * interval_t + xmin_t) && value > (ii * interval_t + xmin_t) ) {
	//	cout << value << " title = " << title_t << endl;
	entries[ii]++;
	//cout << "Adding to " << ii * interval_t + xmin_t << endl;
      }
    }
  }
}
void TopHist::thfill(double value, double weight) { 
  totalentries += weight;
  if(value < xmax_t && value > xmin_t) {       
    // cout << "value " << value << endl;
    for(int ii = 0; ii < bins_t; ii++) { 
      if(value <= ((ii+1) * interval_t + xmin_t) && value > (ii * interval_t + xmin_t) ) {
	//	cout << value << " title = " << title_t << endl;
	//	if(weight > 0) { entries[ii]++;}
	//	if(weight < 0) { entries[ii]--;}

	entries[ii]+= weight;

	//cout << "Adding to " << ii * interval_t + xmin_t << endl;
      }
    }
  }
}

void TopDist::add(string output, bool log) {
  double totnum = 0;
  string titlecase = "";
  string left = "";
  string leftcase = "";

  //  cout << title_t << ", N(attempted) = " << totalentries << ", N(in histo) = " << totnum << endl;
  for(int kk = 0; kk < bins_t; kk++) { 
    entries[kk] = entries[kk]/tot_entries_bin[kk];
  }
  ofstream outf;
  outf.open(output.c_str(), fstream::app);
  outf << "NEW FRAME" << endl;
  outf << "SET WINDOW X 1.6 8 Y 1.6 9" << endl;
  outf << "SET FONT DUPLEX" << endl;
  outf << "TITLE TOP \""    << title_t     << "\"\n";
  outf << "CASE      \""    << titlecase << "\"\n";
  outf << "TITLE LEFT \""   << left      << "\"\n";
  outf << "CASE       \""   << leftcase  << "\"\n";
  outf << "SET INTENSITY 4\n";
  outf << "SET ORDER X Y DX\n";
  if(log) { outf << "SET SCALE Y LOG\n"; }
  outf << "SET LIMITS X " << xmin_t << " " << xmax_t << endl;
  double interval = (xmax_t - xmin_t) / bins_t;
  // cout << interval << endl;
  for(int ii = 0; ii < bins_t; ii++) { 
    outf << ii * interval + xmin_t + interval_t * 0.5 << "\t" << entries[ii] << endl;
  }
  outf << "HIST  " << endl;
  outf.close();
}


void TopHist::add(string output, bool norm, bool log) {
  double totnum = 0;
  string titlecase = "";
  string left = "";
  string leftcase = "";
  for(int jj = 0; jj < bins_t; jj++) { 
    totnum += entries[jj];
  }
  cout << title_t << ", N(attempted) = " << totalentries << ", N(in histo) = " << totnum << endl;
  if(!norm) { totnum = 1; }
  for(int kk = 0; kk < bins_t; kk++) { 
    entries[kk] = entries[kk]/totnum;
  }
  ofstream outf;
  outf.open(output.c_str(), fstream::app);
  outf << "NEW FRAME" << endl;
  outf << "SET WINDOW X 1.6 8 Y 1.6 9" << endl;
  outf << "SET FONT DUPLEX" << endl;
  outf << "TITLE TOP \""    << title_t     << "\"\n";
  outf << "CASE      \""    << titlecase << "\"\n";
  outf << "TITLE LEFT \""   << left      << "\"\n";
  outf << "CASE       \""   << leftcase  << "\"\n";
  outf << "SET INTENSITY 4\n";
  outf << "SET ORDER X Y DX\n";
  if(log) { outf << "SET SCALE Y LOG\n"; }
  outf << "SET LIMITS X " << xmin_t << " " << xmax_t << endl;
  double interval = (xmax_t - xmin_t) / bins_t;
  // cout << interval << endl;
  for(int ii = 0; ii < bins_t; ii++) { 
    outf << ii * interval + xmin_t + interval_t * 0.5 << "\t" << entries[ii] << endl;
  }
  outf << "HIST  " << endl;
  outf.close();
}


void TopHist::add(string output, bool norm, bool log, double normfactor) {
  double totnum = 0;
  string titlecase = "";
  string left = "";
  string leftcase = "";
  for(int jj = 0; jj < bins_t; jj++) { 
    totnum += entries[jj];
  }
  cout << "total entries = " << totalentries << " falling into histo: " << totnum << endl;
  if(!norm) { totnum = 1; }
  for(int kk = 0; kk < bins_t; kk++) { 
    entries[kk] = normfactor * entries[kk]/totnum;
  }
  ofstream outf;
  outf.open(output.c_str(), fstream::app);
  outf << "NEW FRAME" << endl;
  outf << "SET WINDOW X 1.6 8 Y 1.6 9" << endl;
  outf << "SET FONT DUPLEX" << endl;
  outf << "TITLE TOP \""    << title_t     << "\"\n";
  outf << "CASE      \""    << titlecase << "\"\n";
  outf << "TITLE LEFT \""   << left      << "\"\n";
  outf << "CASE       \""   << leftcase  << "\"\n";
  outf << "SET INTENSITY 4\n";
  outf << "SET ORDER X Y DX\n";
  if(log) { outf << "SET SCALE Y LOG\n"; }
  outf << "SET LIMITS X " << xmin_t << " " << xmax_t << endl;
  double interval = (xmax_t - xmin_t) / bins_t;
  // cout << interval << endl;
  for(int ii = 0; ii < bins_t; ii++) { 
    outf << ii * interval + xmin_t + interval_t * 0.5 << "\t" << entries[ii] << endl;
  }
  outf << "HIST  " << endl;
  outf.close();
}


void TopHist::addd(string output, bool norm, bool log, double normfactor) {
  double totnum = 0;
  string titlecase = "";
  string left = "";
  string leftcase = "";
  for(int jj = 0; jj < bins_t; jj++) { 
    totnum += entries[jj];
  }
  if(!norm) { totnum = 1; }
  for(int kk = 0; kk < bins_t; kk++) { 
    entries[kk] = normfactor * bins_t * entries[kk]/ (totnum * (xmax_t - xmin_t) );
  }
  ofstream outf;
  outf.open(output.c_str(), fstream::app);
  outf << "NEW FRAME" << endl;
  outf << "SET WINDOW X 1.6 8 Y 1.6 9" << endl;
  outf << "SET FONT DUPLEX" << endl;
  outf << "TITLE TOP \""    << title_t     << "\"\n";
  outf << "CASE      \""    << titlecase << "\"\n";
  outf << "TITLE LEFT \""   << left      << "\"\n";
  outf << "CASE       \""   << leftcase  << "\"\n";
  outf << "SET INTENSITY 4\n";
  outf << "SET ORDER X Y DX\n";
  if(log) { outf << "SET SCALE Y LOG\n"; }
  outf << "SET LIMITS X " << xmin_t << " " << xmax_t << endl;
  double interval = (xmax_t - xmin_t) / bins_t;
  // cout << interval << endl;
  for(int ii = 0; ii < bins_t; ii++) { 
    outf << ii * interval + xmin_t + interval_t * 0.5 << "\t" << entries[ii] << endl;
  }
  outf << "HIST  " << endl;
  outf.close();
}




double TopHist::getxmax() {
  double totnum = 0;
  string titlecase = "";
  string left = "";
  string leftcase = "";
  double *plotarray = new double[bins_t];
  double maxbin = 0.;
  double xmax = 0.;
  for(int jj = 0; jj < bins_t; jj++) { 
    totnum += entries[jj];
  }
  for(int kk = 0; kk < bins_t; kk++) { 
    plotarray[kk] = bins_t * entries[kk]/ (totnum * (xmax_t - xmin_t) );
  }
  double max = plotarray[0];
  for(int ff = 0; ff < bins_t; ff++) { 
    if(plotarray[ff] > max) { max = plotarray[ff]; maxbin = ff;}
  }
  
  xmax = maxbin * interval_t + xmin_t + interval_t * 0.5;
  return xmax;
}


void TopHist::plotds(bool norm, bool log, double normfactor) {
  double totnum = 0;
  string titlecase = "";
  string left = "";
  string leftcase = "";
  double *plotarray = new double[bins_t];
  for(int jj = 0; jj < bins_t; jj++) { 
    totnum += entries[jj];
  }
  if(!norm) { totnum = 1; }
  cout << "total entries = " << totalentries << " falling into histo: " << totnum << endl;
  for(int kk = 0; kk < bins_t; kk++) { 
    plotarray[kk] = normfactor * bins_t * entries[kk]/ (totalentries * (xmax_t - xmin_t) );
  }
  ofstream outf;
  outf.open(output_t.c_str());
  outf << "NEW FRAME" << endl;
  outf << "SET WINDOW X 1.6 8 Y 1.6 9" << endl;
  outf << "SET FONT DUPLEX" << endl;
  outf << "TITLE TOP \""    << title_t     << "\"\n";
  outf << "CASE      \""    << titlecase << "\"\n";
  outf << "TITLE LEFT \""   << left      << "\"\n";
  outf << "CASE       \""   << leftcase  << "\"\n";
  outf << "SET INTENSITY 4\n";
  outf << "SET ORDER X Y DX\n";
  if(log) { outf << "SET SCALE Y LOG\n"; }
  outf << "SET LIMITS X " << xmin_t << " " << xmax_t << endl;
  // cout << interval << endl;
  for(int ii = 0; ii < bins_t; ii++) { 
    outf << ii * interval_t + xmin_t + interval_t * 0.5 << "\t" << plotarray[ii] << endl;
  }
  outf << "HIST  " << endl;
  outf.close();
}

void TopHist::addds(string output, bool norm, bool log, double normfactor) {
  double totnum = 0;
  string titlecase = "";
  string left = "";
  string leftcase = "";
  for(int jj = 0; jj < bins_t; jj++) { 
    totnum += entries[jj];
  }
  if(!norm) { totnum = 1; }
  cout << "total entries = " << totalentries << " falling into histo: " << totnum << endl;
  for(int kk = 0; kk < bins_t; kk++) { 
    entries[kk] = normfactor * bins_t * entries[kk]/ (totalentries * (xmax_t - xmin_t) );
  }
  ofstream outf;
  outf.open(output.c_str(), fstream::app);
  outf << "NEW FRAME" << endl;
  outf << "SET WINDOW X 1.6 8 Y 1.6 9" << endl;
  outf << "SET FONT DUPLEX" << endl;
  outf << "TITLE TOP \""    << title_t     << "\"\n";
  outf << "CASE      \""    << titlecase << "\"\n";
  outf << "TITLE LEFT \""   << left      << "\"\n";
  outf << "CASE       \""   << leftcase  << "\"\n";
  outf << "SET INTENSITY 4\n";
  outf << "SET ORDER X Y DX\n";
  if(log) { outf << "SET SCALE Y LOG\n"; }
  outf << "SET LIMITS X " << xmin_t << " " << xmax_t << endl;
  double interval = (xmax_t - xmin_t) / bins_t;
  // cout << interval << endl;
  for(int ii = 0; ii < bins_t; ii++) { 
    outf << ii * interval + xmin_t + interval_t * 0.5 << "\t" << entries[ii] << endl;
  }
  outf << "HIST  ~" << endl;
  outf.close();
}


void TopHist::subtract(TopHist one, TopHist two) {
  if( one.xmin_t != two.xmin_t ) { cout << "TopHists do not have the same xmin! " << endl; exit(1); } 
  if( one.xmax_t != two.xmax_t ) { cout << "TopHists do not have the same xmax! " << endl; exit(1); }
  if( one.bins_t != two.bins_t ) { cout << "TopHists do not have the same number of bins! " << endl; exit(1); }


  for(int kk = 0; kk < bins_t; kk++) { 
       entries[kk] = one.entries[kk]-two.entries[kk];
  }

}

void TopHist::add(TopHist one, TopHist two) {
  if( one.xmin_t != two.xmin_t ) { cout << "TopHists do not have the same xmin! " << endl; exit(1); } 
  if( one.xmax_t != two.xmax_t ) { cout << "TopHists do not have the same xmax! " << endl; exit(1); }
  if( one.bins_t != two.bins_t ) { cout << "TopHists do not have the same number of bins! " << endl; exit(1); }
 
  for(int kk = 0; kk < bins_t; kk++) { 
       entries[kk] = one.entries[kk]+two.entries[kk];
  }

}

void TopHist::divide(TopHist one, TopHist two) {
  if( one.xmin_t != two.xmin_t ) { cout << "TopHists do not have the same xmin! " << endl; exit(1); } 
  if( one.xmax_t != two.xmax_t ) { cout << "TopHists do not have the same xmax! " << endl; exit(1); }
  if( one.bins_t != two.bins_t ) { cout << "TopHists do not have the same number of bins! " << endl; exit(1); }


  for(int kk = 0; kk < bins_t; kk++) { 
    if(two.entries[kk]!=0) { 
      entries[kk] = one.entries[kk]/two.entries[kk];
    }
  }

}


double sqr(double x) { 
  return pow(x,2);
}
