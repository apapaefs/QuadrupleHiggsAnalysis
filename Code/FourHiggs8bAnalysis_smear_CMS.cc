#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
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
#include <TNamed.h>
#include <TParameter.h>
#include <TRandom3.h>
#include <TTree.h>

#include "fastjet/PseudoJet.hh"

#include <boost/algorithm/string.hpp>

#include "TopHist.h"
#include "Extended91Observables.h"

using fastjet::PseudoJet;

namespace {

constexpr int kSelectedBJets = 8;
constexpr int kHiggsCount = 4;
constexpr int kHiggsPairCount = kHiggsCount * (kHiggsCount - 1) / 2;
constexpr int kVariableCount = 29;
constexpr double kZBosonMass = 91.1876;
constexpr double kMinimumSmearedEnergy = 1.0e-6;
constexpr unsigned long kJetSmearingSeed = 14101983;
constexpr const char* kExtendedOutputTag = "extended-v2-uniform-smear-v1";
constexpr const char* kLegacyExtendedOutputTag = "extended-v2";
constexpr const char* kJetSmearingModelId = "cms-energy-uniform-fourvector-v1";
constexpr const char* kJetSmearingAcceptanceOrder =
    "raw_abs_eta_then_smear_then_smeared_pt";
constexpr const char* kJetSmearingFourVectorScaling = "uniform_correlated";

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

struct NonBJetCandidate {
  PseudoJet p4;
  bool is_charm = false;
};

using ExtendedReconstruction = extended91::Reconstruction<PseudoJet>;

char* getCmdOption(char** begin, char** end, const std::string& option);
bool cmdOptionExists(char** begin, char** end, const std::string& option);
int parseNonNegativeIntOption(char** begin, char** end, const std::string& option, int default_value);

double deltaR(const PseudoJet& p1, const PseudoJet& p2);
bool overlapsWithAny(const PseudoJet& jet, const std::vector<PseudoJet>& selected, double max_delta_r);
bool jetEfficiencyAccept(const PseudoJet& jet);
double btagWeight(const PseudoJet& jet);
double smearedJetEnergyCMS(const PseudoJet& jet);
PseudoJet smearJetCMSLegacyMassless(const PseudoJet& jet);
PseudoJet smearJetCMSUniformFourVector(const PseudoJet& jet,
                                      double& mass_scaling_residual);

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
void fillExtendedFeatures(const ExtendedReconstruction& reconstruction,
                          double* features,
                          long long& undefined_helicity_count,
                          long long& sanitized_feature_count);
double absoluteHelicityCosine(const PseudoJet& constituent,
                              const PseudoJet& higgs,
                              const PseudoJet& four_higgs,
                              bool& defined);
const std::vector<std::string>& extendedFeatureNames();
const std::vector<std::string>& extendedFeatureUnits();
std::string jsonStringArray(const std::vector<std::string>& values);

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
  rnd.SetSeed(kJetSmearingSeed);

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
  if (tag == std::string("-") + kLegacyExtendedOutputTag) {
    std::cerr << "Error: output tag '" << kLegacyExtendedOutputTag
              << "' belongs to the previous massless smearing model; use '"
              << kExtendedOutputTag << "' for the current analysis" << std::endl;
    return 1;
  }
  const bool write_extended_v2 = tag == std::string("-") + kExtendedOutputTag;
  if (write_extended_v2) {
    std::cout << "Enabling extended-91-v2 Data3 output with smearing model "
              << kJetSmearingModelId << std::endl;
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

  TTree* Data3 = nullptr;
  double extended_features[extended91::kFeatureCount] = {};
  double extended_weight = 0.0;
  Long64_t extended_event_index = -1;
  ULong64_t extended_cut_mask = 0;
  Bool_t passes_legacy_full_selection = false;
  if (write_extended_v2) {
    Data3 = new TTree("Data3", "Extended 91-observable data tree");
    Data3->Branch("features", extended_features, "features[91]/D");
    Data3->Branch("weight", &extended_weight, "weight/D");
    Data3->Branch("event_index", &extended_event_index, "event_index/L");
    Data3->Branch("cut_mask", &extended_cut_mask, "cut_mask/l");
    Data3->Branch("passes_legacy_full_selection", &passes_legacy_full_selection,
                  "passes_legacy_full_selection/O");
  }

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
  double feature_tree_eventcount = 0.0;
  double feature_tree_weight_out = 0.0;
  long long undefined_helicity_count = 0;
  long long sanitized_extended_feature_count = 0;
  long long true_b_upward_pt_migrations = 0;
  long long true_b_downward_pt_migrations = 0;
  long long non_b_upward_pt_migrations = 0;
  long long non_b_downward_pt_migrations = 0;
  long long true_b_upward_pt_migrations_raw_pt_10_12_gev = 0;
  long long true_b_upward_pt_migrations_raw_pt_12_15_gev = 0;
  long long true_b_upward_pt_migrations_raw_pt_15_20_gev = 0;
  long long non_b_upward_pt_migrations_raw_pt_10_12_gev = 0;
  long long non_b_upward_pt_migrations_raw_pt_12_15_gev = 0;
  long long non_b_upward_pt_migrations_raw_pt_15_20_gev = 0;
  double max_smearing_mass_scaling_residual_gev = 0.0;

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

  std::vector<extended91::CanonicalPairing> extended_pairings;
  if (write_extended_v2) {
    extended_pairings = extended91::makeCanonicalPairings();
    if (extended_pairings.size() != static_cast<std::size_t>(extended91::kPairingCount)) {
      std::cerr << "Error: Expected " << extended91::kPairingCount
                << " canonical v2 pairings, generated " << extended_pairings.size() << std::endl;
      return 1;
    }
    if (extendedFeatureNames().size() != static_cast<std::size_t>(extended91::kFeatureCount) ||
        extendedFeatureUnits().size() != static_cast<std::size_t>(extended91::kFeatureCount)) {
      std::cerr << "Error: extended-91-v2 feature metadata does not contain exactly "
                << extended91::kFeatureCount << " entries" << std::endl;
      return 1;
    }
  }

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
    std::vector<NonBJetCandidate> tagged_non_b_candidates;
    if (write_extended_v2) {
      // The versioned v2 path smears every finite, eta-accepted stored jet
      // exactly once before applying the reconstructed-pT threshold.  Both
      // populations are processed before any event-level multiplicity cut so
      // the migration audit is unconditional.
      for (int jj = 0; jj < numbJets; ++jj) {
        PseudoJet bjet_candidate(
            thebJets[1][jj], thebJets[2][jj], thebJets[3][jj], thebJets[0][jj]);
        const double raw_eta = bjet_candidate.eta();
        const double raw_pt = bjet_candidate.perp();
        if (!std::isfinite(raw_eta) || !std::isfinite(raw_pt) ||
            std::fabs(raw_eta) >= kBJetEtaCut) {
          continue;
        }

        double mass_scaling_residual = 0.0;
        PseudoJet smeared = smearJetCMSUniformFourVector(
            bjet_candidate, mass_scaling_residual);
        max_smearing_mass_scaling_residual_gev = std::max(
            max_smearing_mass_scaling_residual_gev, mass_scaling_residual);
        const double smeared_pt = smeared.perp();
        const bool raw_passes_pt = raw_pt > kBJetPtCut;
        const bool smeared_passes_pt =
            std::isfinite(smeared_pt) && smeared_pt > kBJetPtCut;
        // Count every upward migration. The documented 10--20 GeV bins must
        // exhaust this total; a jet stored below 10 GeV therefore breaks the
        // metadata closure and prevents silent reuse of an underspecified input.
        if (!raw_passes_pt && smeared_passes_pt) {
          ++true_b_upward_pt_migrations;
          if (raw_pt >= 10.0 && raw_pt < 12.0) {
            ++true_b_upward_pt_migrations_raw_pt_10_12_gev;
          } else if (raw_pt >= 12.0 && raw_pt < 15.0) {
            ++true_b_upward_pt_migrations_raw_pt_12_15_gev;
          } else if (raw_pt >= 15.0 && raw_pt <= kBJetPtCut) {
            ++true_b_upward_pt_migrations_raw_pt_15_20_gev;
          }
        } else if (raw_passes_pt && !smeared_passes_pt) {
          ++true_b_downward_pt_migrations;
        }
        if (!smeared_passes_pt || !jetEfficiencyAccept(smeared)) {
          continue;
        }
        smeared.set_user_index(static_cast<int>(thebJets[4][jj]));
        true_bjets_unsorted.push_back(smeared);
      }

      for (int jj = 0; jj < numJets; ++jj) {
        PseudoJet jet_candidate(
            theJets[1][jj], theJets[2][jj], theJets[3][jj], theJets[0][jj]);
        const double raw_eta = jet_candidate.eta();
        const double raw_pt = jet_candidate.perp();
        if (!std::isfinite(raw_eta) || !std::isfinite(raw_pt) ||
            std::fabs(raw_eta) >= kBJetEtaCut) {
          continue;
        }

        double mass_scaling_residual = 0.0;
        PseudoJet smeared = smearJetCMSUniformFourVector(
            jet_candidate, mass_scaling_residual);
        max_smearing_mass_scaling_residual_gev = std::max(
            max_smearing_mass_scaling_residual_gev, mass_scaling_residual);
        const double smeared_pt = smeared.perp();
        const bool raw_passes_pt = raw_pt > kBJetPtCut;
        const bool smeared_passes_pt =
            std::isfinite(smeared_pt) && smeared_pt > kBJetPtCut;
        if (!raw_passes_pt && smeared_passes_pt) {
          ++non_b_upward_pt_migrations;
          if (raw_pt >= 10.0 && raw_pt < 12.0) {
            ++non_b_upward_pt_migrations_raw_pt_10_12_gev;
          } else if (raw_pt >= 12.0 && raw_pt < 15.0) {
            ++non_b_upward_pt_migrations_raw_pt_12_15_gev;
          } else if (raw_pt >= 15.0 && raw_pt <= kBJetPtCut) {
            ++non_b_upward_pt_migrations_raw_pt_15_20_gev;
          }
        } else if (raw_passes_pt && !smeared_passes_pt) {
          ++non_b_downward_pt_migrations;
        }
        if (!smeared_passes_pt || !jetEfficiencyAccept(smeared)) {
          continue;
        }
        smeared.set_user_index(static_cast<int>(theJets[4][jj]));
        NonBJetCandidate candidate;
        candidate.p4 = smeared;
        candidate.is_charm = cTag[jj] > 0.0;
        tagged_non_b_candidates.push_back(candidate);
      }
    } else {
      // Preserve the historical untagged preprocessing: require raw pT first,
      // apply the dormant efficiency on the raw jet, then use the massless
      // E'/cosh(eta) mapping documented in the earlier analysis.
      for (int jj = 0; jj < numbJets; ++jj) {
        PseudoJet bjet_candidate(
            thebJets[1][jj], thebJets[2][jj], thebJets[3][jj], thebJets[0][jj]);
        const double raw_eta = bjet_candidate.eta();
        const double raw_pt = bjet_candidate.perp();
        if (!std::isfinite(raw_eta) || !std::isfinite(raw_pt) ||
            raw_pt <= kBJetPtCut || std::fabs(raw_eta) >= kBJetEtaCut ||
            !jetEfficiencyAccept(bjet_candidate)) {
          continue;
        }
        PseudoJet smeared = smearJetCMSLegacyMassless(bjet_candidate);
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
    if (write_extended_v2) {
      for (const NonBJetCandidate& candidate : tagged_non_b_candidates) {
        if (overlapsWithAny(candidate.p4, true_bjets, kDuplicateJetDeltaR)) {
          continue;
        }
        if (candidate.is_charm) {
          c_mistag_candidates_unsorted.push_back(candidate.p4);
        } else {
          light_mistag_candidates_unsorted.push_back(candidate.p4);
        }
      }
    } else {
      for (int jj = 0; jj < numJets; ++jj) {
        PseudoJet jet_candidate(
            theJets[1][jj], theJets[2][jj], theJets[3][jj], theJets[0][jj]);
        const double raw_eta = jet_candidate.eta();
        const double raw_pt = jet_candidate.perp();
        if (!std::isfinite(raw_eta) || !std::isfinite(raw_pt) ||
            raw_pt <= kBJetPtCut || std::fabs(raw_eta) >= kBJetEtaCut ||
            !jetEfficiencyAccept(jet_candidate)) {
          continue;
        }
        PseudoJet smeared = smearJetCMSLegacyMassless(jet_candidate);
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
    const bool pass_bjet_pt_requirement = allAbove(bjet_pts, kSelectedBJetPtCuts);
    if (passed_all_cuts && pass_bjet_pt_requirement) {
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

    const bool pass_chi8_requirement = reco.chi8 < kMaxChi8;
    if (passed_all_cuts && pass_chi8_requirement) {
      pass_chi8 += evweight;
    } else {
      passed_all_cuts = false;
    }

    const bool pass_delta_m_requirement = allBelow(delta_m, kDeltaMCuts);
    if (passed_all_cuts && pass_delta_m_requirement) {
      pass_DeltaM += evweight;
    } else {
      passed_all_cuts = false;
    }

    std::array<double, kHiggsCount> higgs_pts = {};
    for (int i = 0; i < kHiggsCount; ++i) {
      higgs_pts[i] = higgses[i].perp();
    }
    const bool pass_higgs_pt_requirement = allAbove(higgs_pts, kHiggsPtCuts);
    if (passed_all_cuts && pass_higgs_pt_requirement) {
      pass_pthiggses += evweight;
    } else {
      passed_all_cuts = false;
    }

    const bool pass_higgs_drbb_requirement = allBelow(dr_bb, kMaxDeltaRBBInHiggs);
    if (passed_all_cuts && pass_higgs_drbb_requirement) {
      pass_dRbbhiggses += evweight;
    } else {
      passed_all_cuts = false;
    }

    const bool pass_higgs_dr_requirement = allBelow(dr_hh, kMaxDeltaRHiggses);
    if (passed_all_cuts && pass_higgs_dr_requirement) {
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

    if (write_extended_v2) {
      feature_tree_eventcount += 1.0;
      feature_tree_weight_out += evweight;
      const ExtendedReconstruction extended_reconstruction =
          extended91::reconstruct(bjets, extended_pairings, kHiggsMassTargets, kMaxChi8);
      fillExtendedFeatures(extended_reconstruction, extended_features,
                           undefined_helicity_count, sanitized_extended_feature_count);
      extended_weight = evweight;
      extended_event_index = static_cast<Long64_t>(ii);
      extended_cut_mask = 1ULL << 0;
      if (pass_min_drbb) {
        extended_cut_mask |= 1ULL << 1;
      }
      if (pass_bjet_pt_requirement) {
        extended_cut_mask |= 1ULL << 2;
      }
      if (pass_chi8_requirement) {
        extended_cut_mask |= 1ULL << 3;
      }
      if (pass_delta_m_requirement) {
        extended_cut_mask |= 1ULL << 4;
      }
      if (pass_higgs_pt_requirement) {
        extended_cut_mask |= 1ULL << 5;
      }
      if (pass_higgs_drbb_requirement) {
        extended_cut_mask |= 1ULL << 6;
      }
      if (pass_higgs_dr_requirement) {
        extended_cut_mask |= 1ULL << 7;
      }
      if (passed_all_cuts) {
        extended_cut_mask |= 1ULL << 8;
      }
      passes_legacy_full_selection = passed_all_cuts;
      Data3->Fill();
    }

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
  if (write_extended_v2) {
    TNamed analysis_output_tag("analysis_output_tag", kExtendedOutputTag);
    analysis_output_tag.Write();
    TNamed jet_smearing_model_id("jet_smearing_model_id", kJetSmearingModelId);
    jet_smearing_model_id.Write();
    TNamed jet_smearing_acceptance_order("jet_smearing_acceptance_order",
                                         kJetSmearingAcceptanceOrder);
    jet_smearing_acceptance_order.Write();
    TNamed jet_smearing_fourvector_scaling("jet_smearing_fourvector_scaling",
                                           kJetSmearingFourVectorScaling);
    jet_smearing_fourvector_scaling.Write();
    TParameter<Long64_t>("jet_smearing_seed",
                         static_cast<Long64_t>(kJetSmearingSeed)).Write();
    TParameter<double>("jet_smearing_min_energy_gev",
                       kMinimumSmearedEnergy).Write();
    TParameter<int>("jet_smearing_gaussian_draws_per_jet", 1).Write();
    TParameter<int>("jet_smearing_correlated_mass_scaling", 1).Write();
    TParameter<double>("max_smearing_mass_scaling_residual_gev",
                       max_smearing_mass_scaling_residual_gev).Write();
    Data3->Write();
    TNamed observable_schema("Data3_observable_schema", "extended-91-v2");
    observable_schema.Write();
    TNamed feature_names("Data3_feature_names_json",
                         jsonStringArray(extendedFeatureNames()).c_str());
    feature_names.Write();
    TNamed feature_units("Data3_feature_units_json",
                         jsonStringArray(extendedFeatureUnits()).c_str());
    feature_units.Write();
    TNamed cut_mask_definition(
        "Data3_cut_mask_json",
        "{\"0\":\"eight_candidate_population\",\"1\":\"min_dr_bb\","
        "\"2\":\"bjet_pt\",\"3\":\"legacy_chi8\",\"4\":\"legacy_delta_m\","
        "\"5\":\"legacy_higgs_pt\",\"6\":\"legacy_higgs_drbb\","
        "\"7\":\"legacy_higgs_drhh\",\"8\":\"legacy_full_selection\"}");
    cut_mask_definition.Write();
    TParameter<int> feature_count("Data3_feature_count", extended91::kFeatureCount);
    feature_count.Write();
    TParameter<int> pairing_count("Data3_pairing_count", extended91::kPairingCount);
    pairing_count.Write();
  }
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
  const double feature_tree_efficiency =
      total_weight_in != 0.0 ? feature_tree_weight_out / total_weight_in : 0.0;
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
  if (write_extended_v2) {
    std::cout << "feature-tree MC events = \t\t\t\t\t" << feature_tree_eventcount << std::endl;
    std::cout << "feature-tree weight out =\t\t\t\t\t" << feature_tree_weight_out << std::endl;
    std::cout << "feature-tree efficiency =\t\t\t\t\t" << feature_tree_efficiency << std::endl;
    std::cout << "undefined helicity axes =\t\t\t\t\t" << undefined_helicity_count << std::endl;
    std::cout << "sanitized non-finite v2 features =\t\t\t\t" << sanitized_extended_feature_count << std::endl;
    std::cout << "------------------" << std::endl;
  }
  std::cout << "total weight out =\t\t\t\t\t\t" << passcuts << std::endl;
  std::cout << "actual MC events = \t\t\t\t\t\t" << eventcount << std::endl;
  std::cout << "efficiency =\t\t\t\t\t\t\t" << efficiency << std::endl;
  std::cout << "------------------" << std::endl;

  const std::string output_summary = makeOutputName(infile, tag + ".analysis_summary.json");
  std::ofstream outsummary(output_summary.c_str(), std::ios::out);
  if (outsummary) {
    if (write_extended_v2) {
      outsummary << std::setprecision(17);
    }
    outsummary << "{\n";
    outsummary << "  \"input_file\": \"" << infile << "\",\n";
    if (write_extended_v2) {
      outsummary << "  \"analysis_output_tag\": \"" << kExtendedOutputTag << "\",\n";
      outsummary << "  \"jet_smearing_model_id\": \"" << kJetSmearingModelId << "\",\n";
      outsummary << "  \"jet_smearing_acceptance_order\": \""
                 << kJetSmearingAcceptanceOrder << "\",\n";
      outsummary << "  \"jet_smearing_fourvector_scaling\": \""
                 << kJetSmearingFourVectorScaling << "\",\n";
      outsummary << "  \"jet_smearing_correlated_mass_scaling\": true,\n";
      outsummary << "  \"jet_smearing_preserves_jet_mass\": false,\n";
      outsummary << "  \"jet_smearing_gaussian_draws_per_jet\": 1,\n";
      outsummary << "  \"jet_smearing_seed\": " << kJetSmearingSeed << ",\n";
      outsummary << "  \"jet_smearing_min_energy_gev\": "
                 << kMinimumSmearedEnergy << ",\n";
      outsummary << "  \"observable_schema\": \"extended-91-v2\",\n";
    }
    outsummary << "  \"c_mistags\": " << c_mistags << ",\n";
    outsummary << "  \"light_mistags\": " << light_mistags << ",\n";
    outsummary << "  \"required_true_bjets\": " << required_true_bjets << ",\n";
    outsummary << "  \"pt_cut_gev\": " << kBJetPtCut << ",\n";
    outsummary << "  \"eta_cut\": " << kBJetEtaCut << ",\n";
    outsummary << "  \"min_delta_r_jets\": " << kMinDeltaRBJets << ",\n";
    if (write_extended_v2) {
      outsummary << "  \"true_b_upward_pt_migrations\": "
                 << true_b_upward_pt_migrations << ",\n";
      outsummary << "  \"true_b_downward_pt_migrations\": "
                 << true_b_downward_pt_migrations << ",\n";
      outsummary << "  \"non_b_upward_pt_migrations\": "
                 << non_b_upward_pt_migrations << ",\n";
      outsummary << "  \"non_b_downward_pt_migrations\": "
                 << non_b_downward_pt_migrations << ",\n";
      outsummary << "  \"true_b_upward_pt_migrations_raw_pt_10_12_gev\": "
                 << true_b_upward_pt_migrations_raw_pt_10_12_gev << ",\n";
      outsummary << "  \"true_b_upward_pt_migrations_raw_pt_12_15_gev\": "
                 << true_b_upward_pt_migrations_raw_pt_12_15_gev << ",\n";
      outsummary << "  \"true_b_upward_pt_migrations_raw_pt_15_20_gev\": "
                 << true_b_upward_pt_migrations_raw_pt_15_20_gev << ",\n";
      outsummary << "  \"non_b_upward_pt_migrations_raw_pt_10_12_gev\": "
                 << non_b_upward_pt_migrations_raw_pt_10_12_gev << ",\n";
      outsummary << "  \"non_b_upward_pt_migrations_raw_pt_12_15_gev\": "
                 << non_b_upward_pt_migrations_raw_pt_12_15_gev << ",\n";
      outsummary << "  \"non_b_upward_pt_migrations_raw_pt_15_20_gev\": "
                 << non_b_upward_pt_migrations_raw_pt_15_20_gev << ",\n";
      outsummary << "  \"max_smearing_mass_scaling_residual_gev\": "
                 << max_smearing_mass_scaling_residual_gev << ",\n";
    }
    outsummary << "  \"mc_events_in\": " << total_event_in << ",\n";
    outsummary << "  \"total_weight_in\": " << total_weight_in << ",\n";
    outsummary << "  \"preselection_mc_events_out\": " << preselection_eventcount << ",\n";
    outsummary << "  \"preselection_weight_out\": " << pass_ptb << ",\n";
    outsummary << "  \"preselection_efficiency\": " << preselection_efficiency << ",\n";
    if (write_extended_v2) {
      outsummary << "  \"feature_tree_mc_events_out\": " << feature_tree_eventcount << ",\n";
      outsummary << "  \"feature_tree_weight_out\": " << feature_tree_weight_out << ",\n";
      outsummary << "  \"feature_tree_efficiency\": " << feature_tree_efficiency << ",\n";
      outsummary << "  \"undefined_helicity_axis_count\": " << undefined_helicity_count << ",\n";
      outsummary << "  \"sanitized_extended_feature_count\": "
                 << sanitized_extended_feature_count << ",\n";
    }
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

  double epsilon =
      0.75 + (0.95 - 0.75) * (jet.perp() - 20.0) / (50.0 - 20.0);
  epsilon = std::max(0.0, std::min(1.0, epsilon));
  return rnd.Rndm() <= epsilon;
}

double btagWeight(const PseudoJet& /*jet*/) {
  if (kPerfectTagging) {
    return 1.0;
  }
  return 1.0;
}

double smearedJetEnergyCMS(const PseudoJet& jet) {
  const double energy = jet.e();
  if (!std::isfinite(energy) || energy <= 0.0) {
    std::cerr << "Error: cannot smear a jet with non-finite or non-positive energy: "
              << energy << std::endl;
    std::exit(1);
  }

  const double eta = jet.eta();
  double sigma_energy = 0.0;
  if (std::fabs(eta) <= 3.0) {
    sigma_energy = std::sqrt(std::pow(energy * 0.05, 2) + energy * std::pow(1.5, 2));
  } else if (std::fabs(eta) <= 5.0) {
    sigma_energy = std::sqrt(std::pow(energy * 0.130, 2) + energy * std::pow(2.7, 2));
  }

  return std::max(kMinimumSmearedEnergy,
                  energy + rnd.Gaus(0.0, sigma_energy));
}

PseudoJet smearJetCMSLegacyMassless(const PseudoJet& jet) {
  const double smeared_energy = smearedJetEnergyCMS(jet);
  TLorentzVector momentum;
  momentum.SetPtEtaPhiE(smeared_energy / std::cosh(jet.eta()), jet.eta(),
                        jet.phi(), smeared_energy);
  return PseudoJet(momentum.Px(), momentum.Py(), momentum.Pz(), momentum.E());
}

PseudoJet smearJetCMSUniformFourVector(const PseudoJet& jet,
                                      double& mass_scaling_residual) {
  const double energy = jet.e();
  const double smeared_energy = smearedJetEnergyCMS(jet);
  const double scale = smeared_energy / energy;

  // Scale the complete jet four-vector so the existing jet mass receives the
  // same fractional detector response as its energy instead of being set to zero.
  const PseudoJet output(scale * jet.px(), scale * jet.py(), scale * jet.pz(),
                         smeared_energy);
  mass_scaling_residual = std::fabs(output.m() - scale * jet.m());
  if (!std::isfinite(mass_scaling_residual)) {
    std::cerr << "Error: non-finite correlated jet-mass scaling residual"
              << std::endl;
    std::exit(1);
  }
  return output;
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

double absoluteHelicityCosine(const PseudoJet& constituent,
                              const PseudoJet& higgs,
                              const PseudoJet& four_higgs,
                              bool& defined) {
  const extended91::CartesianFourVector constituent_momentum = {
      constituent.px(), constituent.py(), constituent.pz(), constituent.e()};
  const extended91::CartesianFourVector higgs_momentum = {
      higgs.px(), higgs.py(), higgs.pz(), higgs.e()};
  const extended91::CartesianFourVector four_higgs_momentum = {
      four_higgs.px(), four_higgs.py(), four_higgs.pz(), four_higgs.e()};
  return extended91::absoluteHelicityCosine(
      constituent_momentum, higgs_momentum, four_higgs_momentum, defined);
}

void fillExtendedFeatures(const ExtendedReconstruction& reconstruction,
                          double* features,
                          long long& undefined_helicity_count,
                          long long& sanitized_feature_count) {
  std::fill(features, features + extended91::kFeatureCount, 0.0);
  const std::array<PseudoJet, kSelectedBJets>& bjets = reconstruction.canonical_jets;

  PseudoJet four_higgs = reconstruction.higgses[0];
  for (int h = 1; h < kHiggsCount; ++h) {
    four_higgs += reconstruction.higgses[h];
  }

  double ht_8b = 0.0;
  std::vector<std::pair<double, double> > transverse_momenta;
  std::vector<std::pair<double, double> > pt_and_energy;
  transverse_momenta.reserve(kSelectedBJets);
  pt_and_energy.reserve(kSelectedBJets);
  for (int jet_index = 0; jet_index < kSelectedBJets; ++jet_index) {
    features[jet_index] = bjets[jet_index].perp();
    ht_8b += bjets[jet_index].perp();
    transverse_momenta.push_back(std::make_pair(bjets[jet_index].px(), bjets[jet_index].py()));
    pt_and_energy.push_back(std::make_pair(bjets[jet_index].perp(), bjets[jet_index].e()));
  }
  features[8] = std::accumulate(bjets.begin() + 1, bjets.end(), bjets[0]).m();
  features[9] = reconstruction.chi8;

  for (int h = 0; h < kHiggsCount; ++h) {
    features[10 + h] = reconstruction.delta_m[h];
    features[14 + h] = reconstruction.higgses[h].perp();
  }

  int hh_pair_index = 0;
  for (int first = 0; first < kHiggsCount; ++first) {
    for (int second = first + 1; second < kHiggsCount; ++second) {
      features[18 + hh_pair_index] =
          deltaR(reconstruction.higgses[first], reconstruction.higgses[second]);
      ++hh_pair_index;
    }
  }

  std::array<double, kHiggsCount> candidate_masses = {};
  for (int h = 0; h < kHiggsCount; ++h) {
    const extended91::ConstituentPair& pair = reconstruction.pairing[h];
    features[24 + h] = deltaR(bjets[pair[0]], bjets[pair[1]]);
    candidate_masses[h] = reconstruction.higgses[h].m();
    features[28 + h] = candidate_masses[h];
  }
  features[32] = reconstruction.chi8_second;
  features[33] = reconstruction.chi8_second - reconstruction.chi8;
  features[34] = static_cast<double>(reconstruction.n_pairings_chi8_lt60);

  hh_pair_index = 0;
  for (int first = 0; first < kHiggsCount; ++first) {
    for (int second = first + 1; second < kHiggsCount; ++second) {
      const std::array<int, 2> indices = {{first, second}};
      features[35 + hh_pair_index] =
          extended91::subsystemMass(reconstruction.higgses, indices);
      ++hh_pair_index;
    }
  }

  const std::array<std::array<int, 3>, 4> triple_indices = {{
      {{0, 1, 2}}, {{0, 1, 3}}, {{0, 2, 3}}, {{1, 2, 3}},
  }};
  for (int triple = 0; triple < kHiggsCount; ++triple) {
    const std::array<int, 3>& indices = triple_indices[triple];
    features[41 + triple] = extended91::subsystemMass(reconstruction.higgses, indices);
  }

  for (int h = 0; h < kHiggsCount; ++h) {
    const extended91::ConstituentPair& pair = reconstruction.pairing[h];
    const double first_pt = bjets[pair[0]].perp();
    const double second_pt = bjets[pair[1]].perp();
    features[45 + h] = extended91::momentumBalanceFraction(first_pt, second_pt);
  }
  features[49] = extended91::boundedRatio(four_higgs.perp(), four_higgs.m());
  features[50] = std::fabs(four_higgs.rap());
  features[51] = ht_8b;

  const std::array<double, 3> mass_summary = extended91::massSummary(candidate_masses, 125.0);
  features[52] = mass_summary[0];
  features[53] = mass_summary[1];
  features[54] = mass_summary[2];

  for (int h = 0; h < kHiggsCount; ++h) {
    const extended91::ConstituentPair& pair = reconstruction.pairing[h];
    bool helicity_is_defined = false;
    features[55 + h] = absoluteHelicityCosine(
        bjets[pair[0]], reconstruction.higgses[h], four_higgs, helicity_is_defined);
    if (!helicity_is_defined) {
      ++undefined_helicity_count;
    }
    features[59 + h] = std::fabs(bjets[pair[0]].eta() - bjets[pair[1]].eta());
    features[63 + h] =
        extended91::absoluteDeltaPhi(bjets[pair[0]].phi(), bjets[pair[1]].phi());
  }

  std::vector<double> all_pair_delta_r;
  std::vector<double> all_pair_masses;
  all_pair_delta_r.reserve(28);
  all_pair_masses.reserve(28);
  for (int first = 0; first < kSelectedBJets; ++first) {
    for (int second = first + 1; second < kSelectedBJets; ++second) {
      all_pair_delta_r.push_back(deltaR(bjets[first], bjets[second]));
      all_pair_masses.push_back((bjets[first] + bjets[second]).m());
    }
  }
  std::sort(all_pair_delta_r.begin(), all_pair_delta_r.end());
  std::sort(all_pair_masses.begin(), all_pair_masses.end());
  for (int rank = 0; rank < 4; ++rank) {
    features[67 + rank] = all_pair_delta_r[rank];
    features[71 + rank] = all_pair_masses[rank];
  }

  double minimum_higgs_rapidity = reconstruction.higgses[0].rap();
  double maximum_higgs_rapidity = reconstruction.higgses[0].rap();
  for (int h = 1; h < kHiggsCount; ++h) {
    minimum_higgs_rapidity = std::min(minimum_higgs_rapidity, reconstruction.higgses[h].rap());
    maximum_higgs_rapidity = std::max(maximum_higgs_rapidity, reconstruction.higgses[h].rap());
  }
  features[75] = maximum_higgs_rapidity - minimum_higgs_rapidity;

  hh_pair_index = 0;
  for (int first = 0; first < kHiggsCount; ++first) {
    for (int second = first + 1; second < kHiggsCount; ++second) {
      features[76 + hh_pair_index] =
          std::fabs(reconstruction.higgses[first].rap() - reconstruction.higgses[second].rap());
      features[82 + hh_pair_index] = extended91::absoluteDeltaPhi(
          reconstruction.higgses[first].phi(), reconstruction.higgses[second].phi());
      ++hh_pair_index;
    }
  }

  features[88] = extended91::centrality(pt_and_energy);
  features[89] = extended91::transverseSphericity(transverse_momenta);
  features[90] = extended91::minimumMassDistance(all_pair_masses, kZBosonMass);

  for (int feature_index = 0; feature_index < extended91::kFeatureCount; ++feature_index) {
    if (!std::isfinite(features[feature_index])) {
      features[feature_index] = 0.0;
      ++sanitized_feature_count;
    }
  }
}

const std::vector<std::string>& extendedFeatureNames() {
  static const std::vector<std::string> names = {
      "bjet1_pt", "bjet2_pt", "bjet3_pt", "bjet4_pt", "bjet5_pt", "bjet6_pt",
      "bjet7_pt", "bjet8_pt", "m8b", "chi8", "delta_m_h1", "delta_m_h2",
      "delta_m_h3", "delta_m_h4", "higgs1_pt", "higgs2_pt", "higgs3_pt",
      "higgs4_pt", "dr_hh_12", "dr_hh_13", "dr_hh_14", "dr_hh_23", "dr_hh_24",
      "dr_hh_34", "dr_bb_h1", "dr_bb_h2", "dr_bb_h3", "dr_bb_h4", "m_bb_h1",
      "m_bb_h2", "m_bb_h3", "m_bb_h4", "chi8_second", "delta_chi8",
      "n_pairings_chi8_lt60", "m_hh_12", "m_hh_13", "m_hh_14", "m_hh_23",
      "m_hh_24", "m_hh_34", "m_hhh_123", "m_hhh_124", "m_hhh_134", "m_hhh_234",
      "z_bb_h1", "z_bb_h2", "z_bb_h3", "z_bb_h4", "pt_4h_over_m_4h", "abs_y_4h",
      "ht_8b", "mean_m_bb", "std_m_bb", "max_abs_m_bb_minus_125",
      "abs_cos_theta_star_h1", "abs_cos_theta_star_h2", "abs_cos_theta_star_h3",
      "abs_cos_theta_star_h4", "abs_deta_bb_h1", "abs_deta_bb_h2", "abs_deta_bb_h3",
      "abs_deta_bb_h4", "abs_dphi_bb_h1", "abs_dphi_bb_h2", "abs_dphi_bb_h3",
      "abs_dphi_bb_h4", "min_dr_bpair_1", "min_dr_bpair_2", "min_dr_bpair_3",
      "min_dr_bpair_4", "min_m_bpair_1", "min_m_bpair_2", "min_m_bpair_3",
      "min_m_bpair_4", "higgs_rapidity_span", "abs_dy_hh_12", "abs_dy_hh_13",
      "abs_dy_hh_14", "abs_dy_hh_23", "abs_dy_hh_24", "abs_dy_hh_34",
      "abs_dphi_hh_12", "abs_dphi_hh_13", "abs_dphi_hh_14", "abs_dphi_hh_23",
      "abs_dphi_hh_24", "abs_dphi_hh_34", "centrality", "transverse_sphericity", "zness",
  };
  return names;
}

const std::vector<std::string>& extendedFeatureUnits() {
  static const std::vector<std::string> units = {
      "GeV", "GeV", "GeV", "GeV", "GeV", "GeV", "GeV", "GeV", "GeV", "GeV",
      "GeV", "GeV", "GeV", "GeV", "GeV", "GeV", "GeV", "GeV",
      "dimensionless", "dimensionless", "dimensionless", "dimensionless", "dimensionless",
      "dimensionless", "dimensionless", "dimensionless", "dimensionless", "dimensionless",
      "GeV", "GeV", "GeV", "GeV", "GeV", "GeV", "count", "GeV", "GeV", "GeV",
      "GeV", "GeV", "GeV", "GeV", "GeV", "GeV", "GeV", "dimensionless",
      "dimensionless", "dimensionless", "dimensionless", "dimensionless", "dimensionless",
      "GeV", "GeV", "GeV", "GeV", "dimensionless", "dimensionless", "dimensionless",
      "dimensionless", "dimensionless", "dimensionless", "dimensionless", "dimensionless",
      "rad", "rad", "rad", "rad", "dimensionless", "dimensionless", "dimensionless",
      "dimensionless", "GeV", "GeV", "GeV", "GeV", "dimensionless", "dimensionless",
      "dimensionless", "dimensionless", "dimensionless", "dimensionless", "dimensionless",
      "rad", "rad", "rad", "rad", "rad", "rad", "dimensionless", "dimensionless", "GeV",
  };
  return units;
}

std::string jsonStringArray(const std::vector<std::string>& values) {
  std::ostringstream output;
  output << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output << ",";
    }
    output << "\"" << values[index] << "\"";
  }
  output << "]";
  return output.str();
}

}  // namespace
