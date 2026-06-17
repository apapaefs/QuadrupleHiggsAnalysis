#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <TChain.h>
#include <TFile.h>
#include <TLorentzVector.h>
#include <TRandom3.h>
#include <TTree.h>

#include "fastjet/PseudoJet.hh"

#include <boost/algorithm/string.hpp>

#include "TopHist.h"

using fastjet::PseudoJet;

namespace {

constexpr int kSelectedBJets = 8;
constexpr int kHiggsCount = 4;
constexpr int kHiggsPairCount = kHiggsCount * (kHiggsCount - 1) / 2;
constexpr int kVariableCount = 29;

using Pairing = std::array<int, kSelectedBJets>;

const std::array<double, kHiggsCount> kHiggsMassTargets = {{120.0, 115.0, 110.0, 105.0}};
const std::array<double, kSelectedBJets> kSelectedBJetPtCuts = {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}};
const std::array<double, kHiggsCount> kHiggsPtCuts = {{100.0, 100.0, 20.0, 20.0}};
const std::array<double, kHiggsCount> kDeltaMCuts = {{40.0, 50.0, 60.0, 70.0}};

const double kBJetPtCut = 20.0;
const double kBJetEtaCut = 2.5;
const double kMinDeltaRBJets = 0.3;
const double kDuplicateJetDeltaR = 0.05;
const double kMaxChi8 = 60.0;
const double kMaxDeltaRHiggses = 3.5;
const double kMaxDeltaRBBInHiggs = 3.5;

const bool kPerfectTagging = true;
const bool kApplyEfficiency = false;
const bool kResetWeightsToUnity = false;

TChain t("Data");
TRandom3 rnd;

struct Reconstruction {
  double chi8 = std::numeric_limits<double>::infinity();
  Pairing pairing = {};
  std::array<double, kHiggsCount> delta_m = {};
  std::array<PseudoJet, kHiggsCount> higgses = {};
};

char* getCmdOption(char** begin, char** end, const std::string& option);
bool cmdOptionExists(char** begin, char** end, const std::string& option);
int parseNonNegativeIntOption(char** begin, char** end, const std::string& option, int default_value);

double deltaR(const PseudoJet& p1, const PseudoJet& p2);
bool overlapsWithAny(const PseudoJet& jet, const std::vector<PseudoJet>& selected, double max_delta_r);
bool jetEfficiencyAccept(const PseudoJet& jet);
double btagWeight(const PseudoJet& jet);
PseudoJet smearJetCMS(double energy, double pt, double pz, double phi, double eta);

std::string makeOutputName(const std::string& infile, const std::string& replacement);
std::string formatArray(const std::array<double, kHiggsCount>& values);
std::string formatArray(const std::array<double, kSelectedBJets>& values);

void generatePairingsRecursive(std::vector<int>& items, int start, std::vector<Pairing>& pairings);
std::vector<Pairing> makePairings();
Reconstruction findBestReconstruction(const std::vector<PseudoJet>& bjets,
                                      const std::vector<Pairing>& pairings);
std::array<double, kHiggsCount> sortedDeltaM(const std::array<double, kHiggsCount>& delta_m);
std::array<double, kHiggsPairCount> higgsDeltaR(const std::vector<PseudoJet>& higgses);
std::array<double, kHiggsCount> bbDeltaR(const std::vector<PseudoJet>& bjets, const Pairing& pairing);

template <typename T, std::size_t N>
bool allBelow(const std::array<T, N>& values, const std::array<T, N>& cuts) {
  for (std::size_t i = 0; i < N; ++i) {
    if (values[i] >= cuts[i]) {
      return false;
    }
  }
  return true;
}

template <std::size_t N>
bool allBelow(const std::array<double, N>& values, double cut) {
  for (double value : values) {
    if (value >= cut) {
      return false;
    }
  }
  return true;
}

template <std::size_t N>
bool allAbove(const std::array<double, N>& values, const std::array<double, N>& cuts) {
  for (std::size_t i = 0; i < N; ++i) {
    if (values[i] <= cuts[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (!argv[1]) {
    std::cout << "Use: ./HwSimAnalysis [input] [options]" << std::endl;
    return 1;
  }

  const std::string infile(argv[1]);
  rnd.SetSeed(14101983);

  double evweight = 0.0;
  int numbJets = 0;
  int numJets = 0;
  double thebJets[5][100];
  double theJets[5][100];
  double cTag[100];

  t.SetBranchAddress("evweight", &evweight);
  t.SetBranchAddress("thebJets", &thebJets);
  t.SetBranchAddress("numbJets", &numbJets);
  t.SetBranchAddress("theJets", &theJets);
  t.SetBranchAddress("numJets", &numJets);
  t.SetBranchAddress("cTag", &cTag);

  std::string input_path;
  std::ifstream inputlist;
  if (infile.find(".input") != std::string::npos) {
    inputlist.open(infile.c_str());
    if (!inputlist) {
      std::cerr << "Error: Failed to open input file " << infile << std::endl;
      return 1;
    }

    while (inputlist >> input_path) {
      if (!input_path.empty()) {
        t.Add(input_path.c_str());
        std::cout << "Adding " << input_path << std::endl;
      }
    }
  } else if (infile.find(".root") != std::string::npos) {
    std::cout << "Adding " << infile << std::endl;
    t.Add(infile.c_str());
  } else {
    std::cerr << "Error: input must be a .root file or .input file list" << std::endl;
    return 1;
  }

  const int event_number = static_cast<int>(t.GetEntries());
  std::cout << "Total number of events in " << infile << " : " << event_number << std::endl;

  if (kPerfectTagging) {
    std::cout << "WARNING: Perfect b-tagging is enabled" << std::endl;
  }
  if (kResetWeightsToUnity) {
    std::cout << "WARNING: RESETTING ALL WEIGHTS TO = 1" << std::endl;
  }

  bool basic = true;
  if (cmdOptionExists(argv, argv + argc, "-b")) {
    std::cout << "Using previous .evp file and writing .evp2" << std::endl;
    basic = false;
  }

  std::string tag;
  if (cmdOptionExists(argv, argv + argc, "-t")) {
    tag = std::string("-") + getCmdOption(argv, argv + argc, "-t");
    std::cout << "Adding tag: " << tag << std::endl;
  }

  int maxevents = event_number;
  int minevents = 0;
  if (cmdOptionExists(argv, argv + argc, "-n")) {
    maxevents = std::atoi(getCmdOption(argv, argv + argc, "-n"));
  } else if (cmdOptionExists(argv, argv + argc, "-nmax")) {
    maxevents = std::atoi(getCmdOption(argv, argv + argc, "-nmax"));
  }
  if (maxevents > event_number) {
    maxevents = event_number;
  }
  if (maxevents < 1) {
    std::cerr << "Error: maxevents must be at least 1" << std::endl;
    return 1;
  }
  std::cout << "Analyzing up to " << maxevents << std::endl;

  if (cmdOptionExists(argv, argv + argc, "-nmin")) {
    minevents = std::atoi(getCmdOption(argv, argv + argc, "-nmin"));
    if (minevents < 0 || minevents > maxevents) {
      std::cerr << "Error: nmin must be in the range [0, nmax]" << std::endl;
      return 1;
    }
    std::cout << "Analyzing from " << minevents << std::endl;
  }

  const int c_mistags = parseNonNegativeIntOption(argv, argv + argc, "--c-mistags", 0);
  const int light_mistags = parseNonNegativeIntOption(argv, argv + argc, "--light-mistags", 0);
  const int required_true_bjets = kSelectedBJets - c_mistags - light_mistags;
  if (required_true_bjets < 0) {
    std::cerr << "Error: --c-mistags + --light-mistags cannot exceed " << kSelectedBJets << std::endl;
    return 1;
  }
  std::cout << "Candidate composition: " << required_true_bjets << " true b jets, "
            << c_mistags << " c mistag(s), " << light_mistags << " light mistag(s)" << std::endl;

  const std::string output_top = makeOutputName(infile, tag + ".top");
  const std::string output_dat = makeOutputName(infile, tag + ".smearCMS.dat");
  std::ofstream outdat(output_dat.c_str(), std::ios::out);
  if (!outdat) {
    std::cerr << "Error: Cannot open " << output_dat << std::endl;
    return 1;
  }

  std::unordered_set<int> passed_previous;
  if (!basic) {
    const std::string ineventpass = makeOutputName(infile, tag + ".evp");
    std::ifstream inevt(ineventpass.c_str());
    if (!inevt) {
      std::cerr << "Error: Cannot open " << ineventpass << std::endl;
      return 1;
    }

    int event_index = -1;
    while (inevt >> event_index) {
      passed_previous.insert(event_index);
    }
  }

  const std::string outeventpass = makeOutputName(infile, tag + (basic ? ".evp" : ".evp2"));
  std::ofstream outevp(outeventpass.c_str());
  if (!outevp) {
    std::cerr << "Error: Cannot open " << outeventpass << std::endl;
    return 1;
  }

  std::cout << "Preparing Root Tree for event variables" << std::endl;
  const std::string fnameroot = makeOutputName(infile, tag + "_var.smearCMS.root");
  TFile dat2(fnameroot.c_str(), "RECREATE");
  if (dat2.IsZombie()) {
    std::cerr << "Error: Cannot create " << fnameroot << std::endl;
    return 1;
  }

  TTree Data2("Data2", "Data Tree");
  double variables[kVariableCount] = {};
  double weight = 0.0;
  const std::string variable_leaflist = "variables[" + std::to_string(kVariableCount) + "]/D";
  Data2.Branch("variables", variables, variable_leaflist.c_str());
  Data2.Branch("weight", &weight, "weight/D");

  double pass_8b = 0.0;
  double pass_ptb = 0.0;
  double pass_drbb = 0.0;
  double pass_pthiggses = 0.0;
  double pass_chi8 = 0.0;
  double pass_DeltaM = 0.0;
  double pass_dRhiggses = 0.0;
  double pass_dRbbhiggses = 0.0;
  double passcuts = 0.0;
  double eventcount = 0.0;
  double preselection_eventcount = 0.0;
  double total_event_in = 0.0;
  double total_weight_in = 0.0;

  TopHist h_dummy(10, output_top, "dummy histo", 0, 1);
  TopHist h_pT_b(60, output_top, "pT of selected b jets", 0, 300);
  std::array<TopHist, kSelectedBJets> h_pT_b_rank;
  for (int i = 0; i < kSelectedBJets; ++i) {
    h_pT_b_rank[i] = TopHist(60, output_top, "pT of selected b jet " + std::to_string(i + 1), 0, 300);
  }

  TopHist h_chi8(60, output_top, "chi8 min", 0, 300);
  std::array<TopHist, kHiggsCount> h_DeltaM = {{
      TopHist(60, output_top, "Delta M min", 0, 300),
      TopHist(60, output_top, "Delta M med1", 0, 300),
      TopHist(60, output_top, "Delta M med2", 0, 300),
      TopHist(60, output_top, "Delta M max", 0, 300),
  }};
  std::array<TopHist, kHiggsCount> h_pT_higgs;
  for (int i = 0; i < kHiggsCount; ++i) {
    h_pT_higgs[i] = TopHist(60, output_top, "pT of Higgs " + std::to_string(i + 1), 0, 300);
  }
  TopHist h_dR_hh(60, output_top, "Delta R between Higgs bosons", 0, 5);
  TopHist h_dR_bb_higgs(60, output_top, "Delta R between b jets in Higgs candidates", 0, 5);
  TopHist h_m8b(100, output_top, "8b invariant mass", 0, 1500);

  const std::vector<Pairing> pairings = makePairings();
  if (pairings.empty()) {
    std::cerr << "Error: Failed to generate 8b pairings" << std::endl;
    return 1;
  }
  std::cout << "Generated " << pairings.size() << " unique 8b pairings" << std::endl;

  for (int ii = minevents; ii < maxevents; ++ii) {
    if (!basic && passed_previous.find(ii) == passed_previous.end()) {
      continue;
    }

    t.GetEntry(ii);
    if (kResetWeightsToUnity) {
      evweight = 1.0;
    }

    total_weight_in += evweight;
    total_event_in += 1.0;

    std::vector<PseudoJet> true_bjets_unsorted;
    for (int jj = 0; jj < numbJets; ++jj) {
      PseudoJet bjet_candidate(thebJets[1][jj], thebJets[2][jj], thebJets[3][jj], thebJets[0][jj]);
      if (bjet_candidate.perp() > kBJetPtCut &&
          std::fabs(bjet_candidate.eta()) < kBJetEtaCut &&
          jetEfficiencyAccept(bjet_candidate)) {
        PseudoJet smeared = smearJetCMS(bjet_candidate.e(), bjet_candidate.perp(), bjet_candidate.pz(),
                                        bjet_candidate.phi(), bjet_candidate.eta());
        smeared.set_user_index(static_cast<int>(thebJets[4][jj]));
        true_bjets_unsorted.push_back(smeared);
      }
    }

    std::vector<PseudoJet> true_bjets = fastjet::sorted_by_pt(true_bjets_unsorted);
    if (static_cast<int>(true_bjets.size()) < required_true_bjets) {
      continue;
    }
    true_bjets.resize(required_true_bjets);

    std::vector<PseudoJet> c_mistag_candidates_unsorted;
    std::vector<PseudoJet> light_mistag_candidates_unsorted;
    for (int jj = 0; jj < numJets; ++jj) {
      PseudoJet jet_candidate(theJets[1][jj], theJets[2][jj], theJets[3][jj], theJets[0][jj]);
      if (jet_candidate.perp() <= kBJetPtCut ||
          std::fabs(jet_candidate.eta()) >= kBJetEtaCut ||
          !jetEfficiencyAccept(jet_candidate)) {
        continue;
      }

      PseudoJet smeared = smearJetCMS(jet_candidate.e(), jet_candidate.perp(), jet_candidate.pz(),
                                      jet_candidate.phi(), jet_candidate.eta());
      smeared.set_user_index(static_cast<int>(theJets[4][jj]));
      if (overlapsWithAny(smeared, true_bjets, kDuplicateJetDeltaR)) {
        continue;
      }

      if (cTag[jj] > 0.0) {
        c_mistag_candidates_unsorted.push_back(smeared);
      } else {
        light_mistag_candidates_unsorted.push_back(smeared);
      }
    }

    std::vector<PseudoJet> c_mistag_candidates = fastjet::sorted_by_pt(c_mistag_candidates_unsorted);
    if (static_cast<int>(c_mistag_candidates.size()) < c_mistags) {
      continue;
    }
    c_mistag_candidates.resize(c_mistags);

    std::vector<PseudoJet> selected_non_b = c_mistag_candidates;
    std::vector<PseudoJet> light_mistag_candidates;
    for (const PseudoJet& candidate : fastjet::sorted_by_pt(light_mistag_candidates_unsorted)) {
      if (overlapsWithAny(candidate, selected_non_b, kDuplicateJetDeltaR)) {
        continue;
      }
      light_mistag_candidates.push_back(candidate);
      if (static_cast<int>(light_mistag_candidates.size()) == light_mistags) {
        break;
      }
    }
    if (static_cast<int>(light_mistag_candidates.size()) < light_mistags) {
      continue;
    }

    std::vector<PseudoJet> bjets = true_bjets;
    bjets.insert(bjets.end(), c_mistag_candidates.begin(), c_mistag_candidates.end());
    bjets.insert(bjets.end(), light_mistag_candidates.begin(), light_mistag_candidates.end());
    bjets = fastjet::sorted_by_pt(bjets);
    if (static_cast<int>(bjets.size()) < kSelectedBJets) {
      continue;
    }
    bjets.resize(kSelectedBJets);

    for (const PseudoJet& bjet : bjets) {
      evweight *= btagWeight(bjet);
    }
    pass_8b += evweight;

    bool passed_all_cuts = true;

    bool pass_min_drbb = true;
    for (int i = 0; i < kSelectedBJets; ++i) {
      for (int j = i + 1; j < kSelectedBJets; ++j) {
        if (deltaR(bjets[i], bjets[j]) < kMinDeltaRBJets) {
          pass_min_drbb = false;
        }
      }
    }
    if (pass_min_drbb) {
      pass_drbb += evweight;
    } else {
      passed_all_cuts = false;
    }

    std::array<double, kSelectedBJets> bjet_pts = {};
    for (int i = 0; i < kSelectedBJets; ++i) {
      bjet_pts[i] = bjets[i].perp();
    }
    if (passed_all_cuts && allAbove(bjet_pts, kSelectedBJetPtCuts)) {
      pass_ptb += evweight;
      preselection_eventcount += 1.0;
    } else {
      passed_all_cuts = false;
    }

    const Reconstruction reco = findBestReconstruction(bjets, pairings);
    std::vector<PseudoJet> higgses(reco.higgses.begin(), reco.higgses.end());
    higgses = fastjet::sorted_by_pt(higgses);
    const std::array<double, kHiggsCount> delta_m = sortedDeltaM(reco.delta_m);
    const std::array<double, kHiggsPairCount> dr_hh = higgsDeltaR(higgses);
    const std::array<double, kHiggsCount> dr_bb = bbDeltaR(bjets, reco.pairing);

    if (passed_all_cuts && reco.chi8 < kMaxChi8) {
      pass_chi8 += evweight;
    } else {
      passed_all_cuts = false;
    }

    if (passed_all_cuts && allBelow(delta_m, kDeltaMCuts)) {
      pass_DeltaM += evweight;
    } else {
      passed_all_cuts = false;
    }

    std::array<double, kHiggsCount> higgs_pts = {};
    for (int i = 0; i < kHiggsCount; ++i) {
      higgs_pts[i] = higgses[i].perp();
    }
    if (passed_all_cuts && allAbove(higgs_pts, kHiggsPtCuts)) {
      pass_pthiggses += evweight;
    } else {
      passed_all_cuts = false;
    }

    if (passed_all_cuts && allBelow(dr_bb, kMaxDeltaRBBInHiggs)) {
      pass_dRbbhiggses += evweight;
    } else {
      passed_all_cuts = false;
    }

    if (passed_all_cuts && allBelow(dr_hh, kMaxDeltaRHiggses)) {
      pass_dRhiggses += evweight;
    } else {
      passed_all_cuts = false;
    }

    if (passed_all_cuts) {
      passcuts += evweight;
      eventcount += 1.0;
      outevp << ii << std::endl;
    }

    const double m8b = std::accumulate(bjets.begin() + 1, bjets.end(), bjets[0]).m();

    // Variable layout:
    // 0 weight; 1-8 b-jet pT; 9 m8b; 10 chi8; 11-14 DeltaM;
    // 15-18 Higgs pT; 19-24 DeltaR(H,H); 25-28 DeltaR(b,b) in Higgs candidates.
    std::fill(std::begin(variables), std::end(variables), 0.0);
    variables[0] = evweight;
    for (int i = 0; i < kSelectedBJets; ++i) {
      variables[1 + i] = bjet_pts[i];
    }
    variables[9] = m8b;
    variables[10] = reco.chi8;
    for (int i = 0; i < kHiggsCount; ++i) {
      variables[11 + i] = delta_m[i];
      variables[15 + i] = higgs_pts[i];
      variables[25 + i] = dr_bb[i];
    }
    for (int i = 0; i < kHiggsPairCount; ++i) {
      variables[19 + i] = dr_hh[i];
    }

    weight = evweight;
    Data2.Fill();

    for (int i = 0; i < kSelectedBJets; ++i) {
      h_pT_b.thfill(bjets[i].perp(), evweight);
      h_pT_b_rank[i].thfill(bjets[i].perp(), evweight);
    }
    h_chi8.thfill(reco.chi8, evweight);
    for (int i = 0; i < kHiggsCount; ++i) {
      h_DeltaM[i].thfill(delta_m[i], evweight);
      h_pT_higgs[i].thfill(higgs_pts[i], evweight);
      h_dR_bb_higgs.thfill(dr_bb[i], evweight);
    }
    for (double dr : dr_hh) {
      h_dR_hh.thfill(dr, evweight);
    }
    h_m8b.thfill(m8b, evweight);
  }

  dat2.cd();
  Data2.Write();
  dat2.Close();
  std::cout << "A root tree has been written to the file: " << fnameroot << std::endl;

  h_dummy.thfill(0.5);
  h_dummy.plot(false, false);
  h_pT_b.add(output_top, true, false);
  for (TopHist& hist : h_pT_b_rank) {
    hist.add(output_top, true, false);
  }
  h_chi8.add(output_top, true, false);
  for (TopHist& hist : h_DeltaM) {
    hist.add(output_top, true, false);
  }
  for (TopHist& hist : h_pT_higgs) {
    hist.add(output_top, true, false);
  }
  h_dR_hh.add(output_top, true, false);
  h_dR_bb_higgs.add(output_top, true, false);
  h_m8b.add(output_top, true, false);

  const double efficiency = total_weight_in != 0.0 ? passcuts / total_weight_in : 0.0;
  const double preselection_efficiency = total_weight_in != 0.0 ? pass_ptb / total_weight_in : 0.0;
  std::cout << "------------------" << std::endl;
  std::cout << "total weight in =\t\t\t\t\t\t" << total_weight_in << std::endl;
  std::cout << "total MC events in =\t\t\t\t\t\t" << total_event_in << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << "cuts/counters:" << std::endl;
  std::cout << "8bs:\t\t\t\t\t\t\t\t" << pass_8b << std::endl;
  std::cout << "8bs with dR(b,b) > " << kMinDeltaRBJets << "\t\t\t\t\t\t" << pass_drbb << std::endl;
  std::cout << "8bs with pT > " << formatArray(kSelectedBJetPtCuts) << "\t\t\t" << pass_ptb << std::endl;
  std::cout << "chi8 minimum < " << kMaxChi8 << "\t\t\t\t\t\t" << pass_chi8 << std::endl;
  std::cout << "DeltaM sorted < " << formatArray(kDeltaMCuts) << "\t\t\t\t" << pass_DeltaM << std::endl;
  std::cout << "Four reco Higgses with pT > " << formatArray(kHiggsPtCuts) << "\t\t" << pass_pthiggses << std::endl;
  std::cout << "DeltaR(b,b) in reco Higgses < " << kMaxDeltaRBBInHiggs << "\t\t\t\t" << pass_dRbbhiggses << std::endl;
  std::cout << "dR between reco Higgses < " << kMaxDeltaRHiggses << "\t\t\t\t\t" << pass_dRhiggses << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << "preselection MC events = \t\t\t\t\t" << preselection_eventcount << std::endl;
  std::cout << "preselection weight out =\t\t\t\t\t" << pass_ptb << std::endl;
  std::cout << "preselection efficiency =\t\t\t\t\t" << preselection_efficiency << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << "total weight out =\t\t\t\t\t\t" << passcuts << std::endl;
  std::cout << "actual MC events = \t\t\t\t\t\t" << eventcount << std::endl;
  std::cout << "efficiency =\t\t\t\t\t\t\t" << efficiency << std::endl;
  std::cout << "------------------" << std::endl;

  const std::string output_summary = makeOutputName(infile, tag + ".analysis_summary.json");
  std::ofstream outsummary(output_summary.c_str(), std::ios::out);
  if (outsummary) {
    outsummary << "{\n";
    outsummary << "  \"input_file\": \"" << infile << "\",\n";
    outsummary << "  \"c_mistags\": " << c_mistags << ",\n";
    outsummary << "  \"light_mistags\": " << light_mistags << ",\n";
    outsummary << "  \"required_true_bjets\": " << required_true_bjets << ",\n";
    outsummary << "  \"pt_cut_gev\": " << kBJetPtCut << ",\n";
    outsummary << "  \"eta_cut\": " << kBJetEtaCut << ",\n";
    outsummary << "  \"min_delta_r_jets\": " << kMinDeltaRBJets << ",\n";
    outsummary << "  \"mc_events_in\": " << total_event_in << ",\n";
    outsummary << "  \"total_weight_in\": " << total_weight_in << ",\n";
    outsummary << "  \"preselection_mc_events_out\": " << preselection_eventcount << ",\n";
    outsummary << "  \"preselection_weight_out\": " << pass_ptb << ",\n";
    outsummary << "  \"preselection_efficiency\": " << preselection_efficiency << ",\n";
    outsummary << "  \"analysis_mc_events_out\": " << eventcount << ",\n";
    outsummary << "  \"analysis_weight_out\": " << passcuts << ",\n";
    outsummary << "  \"analysis_efficiency\": " << efficiency << "\n";
    outsummary << "}\n";
    std::cout << "Analysis summary JSON written to: " << output_summary << std::endl;
  } else {
    std::cerr << "Warning: Cannot open " << output_summary << " for analysis summary output" << std::endl;
  }

  outdat << efficiency << std::endl;
  return 0;
}

namespace {

char* getCmdOption(char** begin, char** end, const std::string& option) {
  char** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return nullptr;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
  return std::find(begin, end, option) != end;
}

int parseNonNegativeIntOption(char** begin, char** end, const std::string& option, int default_value) {
  if (!cmdOptionExists(begin, end, option)) {
    return default_value;
  }
  char* value = getCmdOption(begin, end, option);
  if (value == nullptr) {
    std::cerr << "Error: missing value for " << option << std::endl;
    std::exit(1);
  }

  const int parsed = std::atoi(value);
  if (parsed < 0) {
    std::cerr << "Error: " << option << " must be non-negative" << std::endl;
    std::exit(1);
  }
  return parsed;
}

double deltaR(const PseudoJet& p1, const PseudoJet& p2) {
  double dphi = p2.phi() - p1.phi();
  if (dphi > M_PI) {
    dphi = 2.0 * M_PI - dphi;
  } else if (dphi < -M_PI) {
    dphi = 2.0 * M_PI + dphi;
  }
  return std::sqrt(std::pow(p1.rap() - p2.rap(), 2) + std::pow(dphi, 2));
}

bool overlapsWithAny(const PseudoJet& jet, const std::vector<PseudoJet>& selected, double max_delta_r) {
  for (const PseudoJet& other : selected) {
    if (deltaR(jet, other) < max_delta_r) {
      return true;
    }
  }
  return false;
}

bool jetEfficiencyAccept(const PseudoJet& jet) {
  if (!kApplyEfficiency) {
    return true;
  }

  double epsilon = 0.75 + (0.95 - 0.75) * jet.perp() / (50.0 - 20.0);
  epsilon = std::max(0.0, std::min(1.0, epsilon));
  return rnd.Rndm() <= epsilon;
}

double btagWeight(const PseudoJet& /*jet*/) {
  if (kPerfectTagging) {
    return 1.0;
  }
  return 1.0;
}

PseudoJet smearJetCMS(double energy, double /*pt*/, double /*pz*/, double phi, double eta) {
  double sigma_energy = 0.0;
  if (std::fabs(eta) <= 3.0) {
    sigma_energy = std::sqrt(std::pow(energy * 0.05, 2) + energy * std::pow(1.5, 2));
  } else if (std::fabs(eta) <= 5.0) {
    sigma_energy = std::sqrt(std::pow(energy * 0.130, 2) + energy * std::pow(2.7, 2));
  }

  const double smeared_energy = std::max(1.0e-6, energy + rnd.Gaus(0.0, sigma_energy));
  TLorentzVector momentum;
  momentum.SetPtEtaPhiE(smeared_energy / std::cosh(eta), eta, phi, smeared_energy);
  return PseudoJet(momentum.Px(), momentum.Py(), momentum.Pz(), momentum.E());
}

std::string makeOutputName(const std::string& infile, const std::string& replacement) {
  std::string output = infile;
  boost::replace_all(output, ".root", replacement);
  boost::replace_all(output, ".input", replacement);
  return output;
}

std::string formatArray(const std::array<double, kHiggsCount>& values) {
  std::ostringstream stream;
  stream << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      stream << ", ";
    }
    stream << values[i];
  }
  stream << "]";
  return stream.str();
}

std::string formatArray(const std::array<double, kSelectedBJets>& values) {
  std::ostringstream stream;
  stream << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      stream << ", ";
    }
    stream << values[i];
  }
  stream << "]";
  return stream.str();
}

void generatePairingsRecursive(std::vector<int>& items, int start, std::vector<Pairing>& pairings) {
  if (items.size() % 2 != 0) {
    return;
  }
  if (start == static_cast<int>(items.size())) {
    Pairing pairing = {};
    std::copy(items.begin(), items.end(), pairing.begin());
    pairings.push_back(pairing);
    return;
  }

  for (int j = start + 1; j < static_cast<int>(items.size()); ++j) {
    std::swap(items[start + 1], items[j]);
    generatePairingsRecursive(items, start + 2, pairings);
    std::swap(items[start + 1], items[j]);
  }
}

std::vector<Pairing> makePairings() {
  std::vector<int> items(kSelectedBJets);
  std::iota(items.begin(), items.end(), 0);

  std::vector<Pairing> pairings;
  generatePairingsRecursive(items, 0, pairings);
  return pairings;
}

Reconstruction findBestReconstruction(const std::vector<PseudoJet>& bjets,
                                      const std::vector<Pairing>& pairings) {
  Reconstruction best;
  for (const Pairing& pairing : pairings) {
    std::array<PseudoJet, kHiggsCount> higgses = {};
    std::array<double, kHiggsCount> delta_m = {};
    double chi8_sum = 0.0;

    for (int h = 0; h < kHiggsCount; ++h) {
      higgses[h] = bjets[pairing[2 * h]] + bjets[pairing[2 * h + 1]];
      delta_m[h] = std::fabs(higgses[h].m() - kHiggsMassTargets[h]);
      chi8_sum += delta_m[h] * delta_m[h];
    }

    const double chi8 = std::sqrt(chi8_sum);
    if (chi8 < best.chi8) {
      best.chi8 = chi8;
      best.pairing = pairing;
      best.delta_m = delta_m;
      best.higgses = higgses;
    }
  }
  return best;
}

std::array<double, kHiggsCount> sortedDeltaM(const std::array<double, kHiggsCount>& delta_m) {
  std::array<double, kHiggsCount> sorted = delta_m;
  std::sort(sorted.begin(), sorted.end());
  return sorted;
}

std::array<double, kHiggsPairCount> higgsDeltaR(const std::vector<PseudoJet>& higgses) {
  std::array<double, kHiggsPairCount> result = {};
  int index = 0;
  for (int i = 0; i < kHiggsCount; ++i) {
    for (int j = i + 1; j < kHiggsCount; ++j) {
      result[index] = deltaR(higgses[i], higgses[j]);
      ++index;
    }
  }
  return result;
}

std::array<double, kHiggsCount> bbDeltaR(const std::vector<PseudoJet>& bjets, const Pairing& pairing) {
  std::array<double, kHiggsCount> result = {};
  for (int h = 0; h < kHiggsCount; ++h) {
    result[h] = deltaR(bjets[pairing[2 * h]], bjets[pairing[2 * h + 1]]);
  }
  return result;
}

}  // namespace
