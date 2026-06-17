#include <TFile.h>
#include <TH1F.h>
#include <iostream>
#include <map>
#include <string>
#include <TRandom3.h>
#include <numeric>


inline double sample_btagscore(
    const std::vector<double>& bin_edges,
    const std::vector<double>& bin_contents,
    TRandom3& rng
) {
    double sum = std::accumulate(
        bin_contents.begin(),
        bin_contents.end(),
        0.0
    );

    if (sum <= 0.0) {
        throw std::runtime_error("Bin contents sum to zero.");
    }

    double r = rng.Uniform(0.0, sum);

    double cumulative = 0.0;
    for (size_t i = 0; i < bin_contents.size(); ++i) {
        cumulative += bin_contents[i];
        if (r <= cumulative) {
            return bin_edges[i];  // same as bin_edges[:-1]
        }
    }

    // Fallback (should not happen)
    return bin_edges.back();
}



inline std::map<
    std::pair<int, int>,
    std::pair<std::vector<double>, std::vector<double>>
> read_btagscore_histograms(const std::string& filePath){
     //std::ifstream file(filePath);
     TFile* root_file = TFile::Open(filePath.c_str());
     //std::map<std::string, TH1F*> btagscore_histograms;

     std::map<
        std::pair<int, int>,
        std::pair<std::vector<double>, std::vector<double>>
     > btagscore_histograms;


     for (int i = 0; i < 6 - 1; ++i) {
        for (int j = 0; j < 19 - 1; ++j) {

            std::string hist_name =
                "hist_eta" + std::to_string(i) +
                "_pt" + std::to_string(j);

            TH1F* histogram =
                dynamic_cast<TH1F*>(root_file->Get(hist_name.c_str()));

            if(histogram) {                
                             int nbins = histogram->GetNbinsX();
                             std::vector<double> bin_edges;
                             std::vector<double> bin_contents;

                             // bin edges: k = 1 .. nbins+1
                             for (int k = 1; k <= nbins + 1; ++k) {
                                  bin_edges.push_back(histogram->GetBinLowEdge(k));
                             }

                             // bin contents: k = 1 .. nbins
                             for (int k = 1; k <= nbins; ++k) {
                                 bin_contents.push_back(histogram->GetBinContent(k));
                              }
			      btagscore_histograms.emplace(
                                           std::make_pair(i, j),
                                           std::make_pair(bin_edges, bin_contents)
                                    );
            } 
	    else {
                   std::cerr << "Warning: Histogram "
                              << hist_name
                              << " not found in the ROOT file."
                              << std::endl;
                 }
        }
    }
    root_file->Close();
    return btagscore_histograms;
}


//function that returns the btagscore for given pt and abs(eta) of a jet:
//
inline double jet_btagscore(double jet_pt, double jet_abs_eta, std::map<std::pair<int, int>,
                                                       std::pair<std::vector<double>, 
						       std::vector<double>>
                                                       >		
		btagscore_histograms){

        float pt_bins[19] = {20,   30,  40,   50,  60,  70,  80, 90, 
	                 100,  120, 140,  160, 180, 200, 250, 300, 
			 400, 600, 1000};
        float abs_eta_bins[6] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5};

	int pt_bin_index  = -1;
        int eta_bin_index = -1;

        // pt bin
        for (int i = 0; i < sizeof(pt_bins)/sizeof(pt_bins[0]) - 1; ++i) {
           if (pt_bins[i] <= jet_pt && jet_pt < pt_bins[i + 1]) {
              pt_bin_index = i;
              break;
           }
        }
// abs(eta) bin
        for (int i = 0; i < sizeof(abs_eta_bins)/sizeof(abs_eta_bins[0])  - 1; ++i) {
            if (abs_eta_bins[i] <= jet_abs_eta &&
               jet_abs_eta < abs_eta_bins[i + 1]) {
               eta_bin_index = i;
               break;
            }
        }

       if (pt_bin_index == -1 || eta_bin_index == -1) {
           throw std::invalid_argument("Jet pt or abs(eta) is out of the defined bin ranges.");
       }

       auto it = btagscore_histograms.find(
                 std::make_pair(eta_bin_index, pt_bin_index)
                 );

       if (it == btagscore_histograms.end()) {
                 throw std::runtime_error(
                 "No histogram found for the given pt and abs(eta) bin.");
        }

        TRandom3 rng(0);

        const std::vector<double>& bin_edges    = it->second.first;
        const std::vector<double>& bin_contents = it->second.second;
        double btagscore = sample_btagscore(bin_edges, bin_contents, rng);
        return btagscore;
}


/*int main(){
	  std::cout << "Hello, world!" << std::endl;

	  std::string filePath_NonB = "../btagscore/JetBtagDeepFlavB_NonB_Distributions.root";

          std::string filePath_B ="../btagscore/JetBtagDeepFlavB_B_Distributions.root";

          std::map<
                 std::pair<int, int>,
                 std::pair<std::vector<double>, std::vector<double>>
                 >   JetBtagDeepFlavB_B_Distributions = read_btagscore_histograms(filePath_B);

	  std::map<
                 std::pair<int, int>,
                 std::pair<std::vector<double>, std::vector<double>>
                 >   JetBtagDeepFlavB_NonB_Distributions = read_btagscore_histograms(filePath_NonB);


	  double jet_pt = 85;
          double jet_abs_eta = 1.2;


	  double btagscore_B= jet_btagscore(jet_pt, jet_abs_eta, 
			                    JetBtagDeepFlavB_B_Distributions);

	  double btagscore_nonB = jet_btagscore(jet_pt, jet_abs_eta, JetBtagDeepFlavB_NonB_Distributions);

	  std::cout<<" btagscore_B="<<btagscore_B<<std::endl;
	  std::cout<<" btagscore_nonB="<<btagscore_nonB<<std::endl;

    }

*/
