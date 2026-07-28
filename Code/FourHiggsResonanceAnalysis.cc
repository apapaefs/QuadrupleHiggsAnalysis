#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <TChain.h>
#include <TFile.h>
#include <TLorentzVector.h>
#include <TNamed.h>
#include <TParameter.h>
#include <TRandom3.h>
#include <TTree.h>

namespace {

constexpr int kMaxInputJets = 100;
constexpr int kHiggsCount = 4;
constexpr int kPairCount = 6;
constexpr int kPaddedJetCount = 8;
constexpr int kDefaultMaxRecoJets = 10;
constexpr int kHardMaxRecoJets = 10;
constexpr double kBJetPtCut = 20.0;
constexpr double kBJetEtaCut = 2.5;
constexpr double kDuplicateJetDeltaR = 0.05;
constexpr double kHiggsMass = 125.0;
constexpr const char* kMethodVersion = "resonance-hybrid-v1.2-uniform-fourvector-smearing";
constexpr const char* kPreprocessingVersion = "resonance-preprocessing-v2";
constexpr const char* kSmearingModelId = "cms-energy-uniform-fourvector-v1";

enum JetSource {
  kTrueB = 0,
  kCharmMistag = 1,
  kLightMistag = 2,
};

struct Options {
  std::string input;
  std::string output_root;
  std::string output_json;
  Long64_t max_events = -1;
  int max_reco_jets = kDefaultMaxRecoJets;
  int c_mistags = 0;
  int light_mistags = 0;
  unsigned int seed = 14101983U;
  bool smear = true;
};

struct Counter {
  Long64_t events = 0;
  double sumw = 0.0;
  double sumw2 = 0.0;

  void fill(double weight) {
    ++events;
    sumw += weight;
    sumw2 += weight * weight;
  }
};

struct Jet {
  TLorentzVector p4;
  int tag_multiplicity = 1;
  int source = kTrueB;
  int original_index = -1;
};

struct Candidate {
  TLorentzVector p4;
  int type = 0;  // 0: resolved pair, 1: merged double-b jet.
  int first_jet = -1;
  int second_jet = -1;
};

struct BestReconstruction {
  bool valid = false;
  double best_score = std::numeric_limits<double>::infinity();
  double second_score = std::numeric_limits<double>::infinity();
  Long64_t configurations = 0;
  std::array<Candidate, kHiggsCount> candidates = {};

  void consider(const std::array<Candidate, kHiggsCount>& proposal) {
    double score = 0.0;
    for (const Candidate& candidate : proposal) {
      const double residual = (candidate.p4.M() - kHiggsMass) / kHiggsMass;
      score += residual * residual;
    }

    ++configurations;
    if (!valid || score < best_score) {
      second_score = best_score;
      best_score = score;
      candidates = proposal;
      valid = true;
    } else if (score < second_score) {
      second_score = score;
    }
  }
};

using IndexCallback = std::function<void(const std::vector<int>&)>;
using PairList = std::vector<std::array<int, 2> >;
using PairingCallback = std::function<void(const PairList&)>;

bool endsWith(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string withoutExtension(const std::string& path) {
  const std::size_t slash = path.find_last_of("/\\");
  const std::size_t dot = path.find_last_of('.');
  if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
    return path;
  }
  return path.substr(0, dot);
}

std::string jsonEscape(const std::string& value) {
  std::ostringstream output;
  for (char character : value) {
    switch (character) {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default: output << character; break;
    }
  }
  return output.str();
}

long long parseLongLong(const char* value, const std::string& option) {
  if (value == nullptr) {
    throw std::runtime_error("missing value for " + option);
  }
  char* end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  if (end == value || *end != '\0') {
    throw std::runtime_error("invalid integer for " + option + ": " + value);
  }
  return parsed;
}

void printUsage(const char* executable) {
  std::cerr
      << "Usage: " << executable << " INPUT.root|INPUT.input [options]\n"
      << "Options:\n"
      << "  --output FILE.root       Output ROOT feature file\n"
      << "  --max-events N           Process at most N events\n"
      << "  --max-reco-jets N        Use the leading N accepted true-b jets (4-10; default 10)\n"
      << "  --c-mistags N            Require exactly N charm mistag objects (default 0)\n"
      << "  --light-mistags N        Require exactly N light mistag objects (default 0)\n"
      << "  --seed N                 Deterministic smearing seed (default 14101983)\n"
      << "  --no-smear               Disable CMS-style jet-energy smearing\n";
}

Options parseOptions(int argc, char* argv[]) {
  if (argc < 2) {
    printUsage(argv[0]);
    throw std::runtime_error("missing input file");
  }
  if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    printUsage(argv[0]);
    std::exit(0);
  }

  Options options;
  options.input = argv[1];
  if (!endsWith(options.input, ".root") && !endsWith(options.input, ".input")) {
    throw std::runtime_error("input must end in .root or .input");
  }

  for (int argument = 2; argument < argc; ++argument) {
    const std::string option = argv[argument];
    if (option == "--no-smear") {
      options.smear = false;
    } else if (option == "--output") {
      if (++argument >= argc) throw std::runtime_error("missing value for --output");
      options.output_root = argv[argument];
    } else if (option == "--max-events") {
      if (++argument >= argc) throw std::runtime_error("missing value for --max-events");
      options.max_events = parseLongLong(argv[argument], "--max-events");
    } else if (option == "--max-reco-jets") {
      if (++argument >= argc) throw std::runtime_error("missing value for --max-reco-jets");
      options.max_reco_jets = static_cast<int>(parseLongLong(argv[argument], "--max-reco-jets"));
    } else if (option == "--c-mistags") {
      if (++argument >= argc) throw std::runtime_error("missing value for --c-mistags");
      options.c_mistags = static_cast<int>(parseLongLong(argv[argument], "--c-mistags"));
    } else if (option == "--light-mistags") {
      if (++argument >= argc) throw std::runtime_error("missing value for --light-mistags");
      options.light_mistags = static_cast<int>(parseLongLong(argv[argument], "--light-mistags"));
    } else if (option == "--seed") {
      if (++argument >= argc) throw std::runtime_error("missing value for --seed");
      const long long seed = parseLongLong(argv[argument], "--seed");
      if (seed < 0 || seed > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("--seed is outside the unsigned-integer range");
      }
      options.seed = static_cast<unsigned int>(seed);
    } else if (option == "--help" || option == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }

  if (options.max_events == 0 || options.max_events < -1) {
    throw std::runtime_error("--max-events must be positive");
  }
  if (options.max_reco_jets < 4 || options.max_reco_jets > kHardMaxRecoJets) {
    throw std::runtime_error("--max-reco-jets must be in the range [4, 10]");
  }
  if (options.c_mistags < 0 || options.light_mistags < 0 ||
      options.c_mistags + options.light_mistags > 8) {
    throw std::runtime_error("mistag counts must be non-negative and sum to at most 8");
  }

  if (options.output_root.empty()) {
    options.output_root = withoutExtension(options.input) + "_resonance_features.root";
  } else if (!endsWith(options.output_root, ".root")) {
    options.output_root += ".root";
  }
  options.output_json = withoutExtension(options.output_root) + ".analysis_summary.json";
  return options;
}

std::vector<std::string> inputFiles(const std::string& input) {
  if (endsWith(input, ".root")) {
    return {input};
  }

  std::ifstream list(input.c_str());
  if (!list) {
    throw std::runtime_error("cannot open input list " + input);
  }

  std::vector<std::string> files;
  std::string line;
  while (std::getline(list, line)) {
    const std::size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos || line[first] == '#') continue;
    std::istringstream parser(line.substr(first));
    std::string path;
    parser >> path;
    if (!path.empty()) files.push_back(path);
  }
  if (files.empty()) {
    throw std::runtime_error("input list contains no ROOT files: " + input);
  }
  return files;
}

void validateInputFile(const std::string& path, bool need_mistag_branches) {
  TFile input(path.c_str(), "READ");
  if (input.IsZombie()) {
    throw std::runtime_error("cannot open ROOT input " + path);
  }
  TTree* tree = dynamic_cast<TTree*>(input.Get("Data"));
  if (tree == nullptr) {
    throw std::runtime_error("ROOT input does not contain a Data tree: " + path);
  }

  const std::array<const char*, 4> required = {
      "evweight", "thebJets", "numbJets", "bHadronMultiplicity"};
  for (const char* branch : required) {
    if (tree->GetBranch(branch) == nullptr) {
      if (std::string(branch) == "bHadronMultiplicity") {
        throw std::runtime_error(
            "Data tree in " + path +
            " does not contain bHadronMultiplicity; regenerate it with the updated HwSim library");
      }
      throw std::runtime_error("Data tree in " + path + " is missing required branch " + branch);
    }
  }

  if (need_mistag_branches) {
    const std::array<const char*, 3> mistag_required = {"theJets", "numJets", "cTag"};
    for (const char* branch : mistag_required) {
      if (tree->GetBranch(branch) == nullptr) {
        throw std::runtime_error(
            "Data tree in " + path + " is missing " + branch +
            ", which is required when c/light mistags are requested");
      }
    }
  }
}

double absoluteDeltaPhi(double first, double second) {
  double difference = std::fmod(first - second, 2.0 * M_PI);
  if (difference > M_PI) difference -= 2.0 * M_PI;
  if (difference < -M_PI) difference += 2.0 * M_PI;
  return std::fabs(difference);
}

double deltaR(const TLorentzVector& first, const TLorentzVector& second) {
  const double dy = first.Rapidity() - second.Rapidity();
  const double dphi = absoluteDeltaPhi(first.Phi(), second.Phi());
  return std::sqrt(dy * dy + dphi * dphi);
}

bool overlaps(const TLorentzVector& candidate, const std::vector<Jet>& jets) {
  for (const Jet& jet : jets) {
    if (deltaR(candidate, jet.p4) < kDuplicateJetDeltaR) return true;
  }
  return false;
}

TLorentzVector smearJetCMSUniformFourVector(const TLorentzVector& input,
                                           TRandom3& random,
                                           bool enabled,
                                           double& mass_scaling_residual) {
  if (!enabled) {
    mass_scaling_residual = 0.0;
    return input;
  }

  const double energy = input.E();
  // Uniform four-vector scaling divides by the input energy, so reject an
  // invalid event record rather than silently producing non-finite features.
  if (!std::isfinite(energy) || energy <= 0.0) {
    throw std::runtime_error("cannot smear a jet with non-finite or non-positive energy");
  }
  const double eta = input.Eta();
  double sigma_energy = 0.0;
  if (std::fabs(eta) <= 3.0) {
    sigma_energy = std::sqrt(std::pow(0.05 * energy, 2) + energy * std::pow(1.5, 2));
  } else if (std::fabs(eta) <= 5.0) {
    sigma_energy = std::sqrt(std::pow(0.130 * energy, 2) + energy * std::pow(2.7, 2));
  }

  // Draw exactly one energy fluctuation and apply the same factor to all four
  // components.  The jet direction is unchanged and its mass scales in
  // correlation with the energy, m'=(E'/E)m.
  const double delta_energy = random.Gaus(0.0, sigma_energy);
  const double smeared_energy = std::max(1.0e-6, energy + delta_energy);
  const double scale = smeared_energy / energy;
  TLorentzVector output;
  output.SetPxPyPzE(scale * input.Px(), scale * input.Py(), scale * input.Pz(),
                   smeared_energy);
  mass_scaling_residual = std::fabs(output.M() - scale * input.M());
  return output;
}

int upwardMigrationRawPtBin(double raw_pt) {
  if (raw_pt >= 10.0 && raw_pt < 12.0) return 0;
  if (raw_pt >= 12.0 && raw_pt < 15.0) return 1;
  if (raw_pt >= 15.0 && raw_pt <= kBJetPtCut) return 2;
  return -1;
}

void combinationsRecursive(const std::vector<int>& values,
                           int needed,
                           std::size_t start,
                           std::vector<int>& selected,
                           const IndexCallback& callback) {
  if (needed == 0) {
    callback(selected);
    return;
  }
  if (needed < 0 || values.size() - start < static_cast<std::size_t>(needed)) return;

  for (std::size_t index = start;
       index + static_cast<std::size_t>(needed) <= values.size(); ++index) {
    selected.push_back(values[index]);
    combinationsRecursive(values, needed - 1, index + 1, selected, callback);
    selected.pop_back();
  }
}

void combinations(const std::vector<int>& values, int needed, const IndexCallback& callback) {
  std::vector<int> selected;
  combinationsRecursive(values, needed, 0, selected, callback);
}

void pairingsRecursive(const std::vector<int>& remaining,
                       PairList& pairs,
                       const PairingCallback& callback) {
  if (remaining.empty()) {
    callback(pairs);
    return;
  }

  const int first = remaining.front();
  for (std::size_t partner = 1; partner < remaining.size(); ++partner) {
    std::vector<int> next;
    next.reserve(remaining.size() - 2);
    for (std::size_t index = 1; index < remaining.size(); ++index) {
      if (index != partner) next.push_back(remaining[index]);
    }
    pairs.push_back({{first, remaining[partner]}});
    pairingsRecursive(next, pairs, callback);
    pairs.pop_back();
  }
}

void pairings(const std::vector<int>& selected, const PairingCallback& callback) {
  PairList pairs;
  pairingsRecursive(selected, pairs, callback);
}

BestReconstruction reconstruct(const std::vector<Jet>& jets,
                               const std::vector<int>& fixed_mistag_indices,
                               int c_mistags,
                               int light_mistags) {
  std::vector<int> true_single_indices;
  std::vector<int> double_b_indices;
  for (std::size_t index = 0; index < jets.size(); ++index) {
    if (jets[index].source != kTrueB) continue;
    if (jets[index].tag_multiplicity >= 2) {
      double_b_indices.push_back(static_cast<int>(index));
    } else {
      true_single_indices.push_back(static_cast<int>(index));
    }
  }

  BestReconstruction best;
  const int mistag_count = c_mistags + light_mistags;
  for (int n_merged = 0; n_merged <= kHiggsCount; ++n_merged) {
    const int n_resolved = kHiggsCount - n_merged;
    const int required_true_singles = 2 * n_resolved - mistag_count;
    if (required_true_singles < 0 ||
        n_merged > static_cast<int>(double_b_indices.size()) ||
        required_true_singles > static_cast<int>(true_single_indices.size())) {
      continue;
    }

    combinations(double_b_indices, n_merged, [&](const std::vector<int>& selected_double_b) {
      combinations(true_single_indices, required_true_singles,
                   [&](const std::vector<int>& selected_true_single) {
        std::vector<int> selected_single = selected_true_single;
        selected_single.insert(selected_single.end(), fixed_mistag_indices.begin(),
                               fixed_mistag_indices.end());
        if (static_cast<int>(selected_single.size()) != 2 * n_resolved) return;

        pairings(selected_single, [&](const PairList& resolved_pairs) {
          std::array<Candidate, kHiggsCount> proposal = {};
          int candidate_index = 0;
          for (int jet_index : selected_double_b) {
            Candidate& candidate = proposal[candidate_index++];
            candidate.p4 = jets[jet_index].p4;
            candidate.type = 1;
            candidate.first_jet = jet_index;
            candidate.second_jet = -1;
          }
          for (const std::array<int, 2>& pair : resolved_pairs) {
            Candidate& candidate = proposal[candidate_index++];
            candidate.p4 = jets[pair[0]].p4 + jets[pair[1]].p4;
            candidate.type = 0;
            candidate.first_jet = pair[0];
            candidate.second_jet = pair[1];
          }
          if (candidate_index == kHiggsCount) best.consider(proposal);
        });
      });
    });
  }
  return best;
}

double transverseSphericity(const std::vector<Jet>& jets) {
  double xx = 0.0;
  double xy = 0.0;
  double yy = 0.0;
  for (const Jet& jet : jets) {
    xx += jet.p4.Px() * jet.p4.Px();
    xy += jet.p4.Px() * jet.p4.Py();
    yy += jet.p4.Py() * jet.p4.Py();
  }
  const double trace = xx + yy;
  if (trace <= 0.0) return 0.0;
  const double discriminant = std::sqrt(std::max(0.0, (xx - yy) * (xx - yy) + 4.0 * xy * xy));
  const double smaller_eigenvalue = 0.5 * (trace - discriminant);
  return std::max(0.0, std::min(1.0, 2.0 * smaller_eigenvalue / trace));
}

double centrality(const std::vector<Jet>& jets) {
  double sum_pt = 0.0;
  double sum_energy = 0.0;
  for (const Jet& jet : jets) {
    sum_pt += jet.p4.Pt();
    sum_energy += jet.p4.E();
  }
  return sum_energy > 0.0 ? sum_pt / sum_energy : 0.0;
}

std::vector<std::string> featureNames() {
  std::vector<std::string> names = {
      "n_merged", "category", "best_score", "second_score", "score_gap"};
  for (int index = 1; index <= kPaddedJetCount; ++index) {
    names.push_back("jet_pt_" + std::to_string(index));
  }
  const std::array<std::string, 8> candidate_fields = {
      "e", "px", "py", "pz", "mass", "pt", "y", "type"};
  for (const std::string& field : candidate_fields) {
    for (int index = 1; index <= kHiggsCount; ++index) {
      names.push_back("higgs_" + field + "_" + std::to_string(index));
    }
  }
  const std::array<std::string, kPairCount> pair_labels = {
      "12", "13", "14", "23", "24", "34"};
  const std::array<std::string, 4> pair_fields = {"mass", "dr", "dy", "dphi"};
  for (const std::string& field : pair_fields) {
    for (const std::string& label : pair_labels) {
      names.push_back("pair_" + field + "_" + label);
    }
  }
  names.insert(names.end(), {"m4h", "pt4h", "y4h", "ht", "centrality", "sphericity"});
  return names;
}

std::string jsonStringArray(const std::vector<std::string>& values) {
  std::ostringstream output;
  output << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ",";
    output << "\"" << jsonEscape(values[index]) << "\"";
  }
  output << "]";
  return output.str();
}

const char* branchSchemaJson() {
  return R"json({
"tree":"ResonanceFeatures",
"schema":"resonance-hybrid-v1",
"candidate_order":"descending_pt",
"jet_pt_definition":"pT-ordered jets used by the best reconstruction, zero padded to eight",
"branches":{
"event_index":"Long64_t","weight":"Double_t raw generator weight; no tag efficiency applied",
"raw_bjets":"Int_t","accepted_bjets":"Int_t true-b jets","accepted_single_bjets":"Int_t true-b jets with capped multiplicity one","accepted_merged_bjets":"Int_t true-b jets with capped multiplicity two",
"accepted_cjet_candidates":"Int_t","accepted_lightjet_candidates":"Int_t","accepted_tag_equivalents":"Int_t over the bounded true-b pool plus required mistags",
"reco_jets_considered":"Int_t","reco_jets_used":"Int_t","n_configurations":"Long64_t",
"n_true_single":"Int_t","n_double_b":"Int_t","n_c_mistag":"Int_t","n_light_mistag":"Int_t",
"n_merged":"Int_t [0,4]","category":"Int_t 0=resolved,1=mixed,2=boosted",
"best_score":"Double_t","second_score":"Double_t; -1 if unavailable","score_gap":"Double_t; -1 if unavailable",
"jet_pt":"Double_t[8]","higgs_e":"Double_t[4]","higgs_px":"Double_t[4]","higgs_py":"Double_t[4]","higgs_pz":"Double_t[4]","higgs_mass":"Double_t[4]","higgs_pt":"Double_t[4]","higgs_y":"Double_t[4]","higgs_type":"Int_t[4] 0=resolved,1=merged",
"higgs_constituent1":"Int_t[4] index in bounded candidate pool","higgs_constituent2":"Int_t[4]; -1 for merged",
"higgs_constituent1_source":"Int_t[4] 0=true-b,1=charm,2=light","higgs_constituent2_source":"Int_t[4]; -1 for merged",
"pair_mass":"Double_t[6]","pair_dr":"Double_t[6]","pair_dy":"Double_t[6]","pair_dphi":"Double_t[6]",
"m4h":"Double_t","pt4h":"Double_t","y4h":"Double_t","ht":"Double_t selected reconstruction jets","centrality":"Double_t selected reconstruction jets","sphericity":"Double_t selected reconstruction jets"
}})json";
}

void writeCounter(std::ostream& output, const Counter& counter, int indent) {
  const std::string spaces(static_cast<std::size_t>(indent), ' ');
  output << spaces << "{\"events\": " << counter.events
         << ", \"sumw\": " << counter.sumw
         << ", \"sumw2\": " << counter.sumw2 << "}";
}

double fraction(double numerator, double denominator) {
  return denominator != 0.0 ? numerator / denominator : 0.0;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const Options options = parseOptions(argc, argv);
    const bool need_mistag_branches = options.c_mistags + options.light_mistags > 0;
    const std::vector<std::string> files = inputFiles(options.input);

    TChain chain("Data");
    for (const std::string& path : files) {
      validateInputFile(path, need_mistag_branches);
      if (chain.Add(path.c_str()) == 0) {
        throw std::runtime_error("failed to add ROOT input " + path);
      }
      std::cout << "Adding " << path << "\n";
    }

    const Long64_t available_events = chain.GetEntries();
    if (available_events <= 0) {
      throw std::runtime_error("no entries found in input Data tree");
    }
    const Long64_t events_to_run =
        options.max_events > 0 ? std::min(options.max_events, available_events) : available_events;

    double event_weight = 0.0;
    int number_bjets = 0;
    int number_jets = 0;
    double bjets[5][kMaxInputJets] = {};
    double jets[5][kMaxInputJets] = {};
    double charm_tag[kMaxInputJets] = {};
    int b_hadron_multiplicity[kMaxInputJets] = {};
    chain.SetBranchAddress("evweight", &event_weight);
    chain.SetBranchAddress("thebJets", &bjets);
    chain.SetBranchAddress("numbJets", &number_bjets);
    chain.SetBranchAddress("bHadronMultiplicity", &b_hadron_multiplicity);
    if (need_mistag_branches) {
      chain.SetBranchAddress("theJets", &jets);
      chain.SetBranchAddress("numJets", &number_jets);
      chain.SetBranchAddress("cTag", &charm_tag);
    }

    TFile output(options.output_root.c_str(), "RECREATE");
    if (output.IsZombie()) {
      throw std::runtime_error("cannot create output ROOT file " + options.output_root);
    }
    TTree tree("ResonanceFeatures", "Hybrid resolved/merged four-Higgs feature tree");

    Long64_t event_index = -1;
    double weight = 0.0;
    int raw_bjets = 0;
    int accepted_bjets = 0;
    int accepted_single_bjets = 0;
    int accepted_merged_bjets = 0;
    int accepted_cjet_candidates = 0;
    int accepted_lightjet_candidates = 0;
    int accepted_tag_equivalents = 0;
    int reco_jets_considered = 0;
    int reco_jets_used = 0;
    Long64_t n_configurations = 0;
    int n_true_single = 0;
    int n_double_b = 0;
    int n_c_mistag = options.c_mistags;
    int n_light_mistag = options.light_mistags;
    int n_merged = 0;
    int category = -1;
    double best_score = 0.0;
    double second_score = -1.0;
    double score_gap = -1.0;
    double jet_pt[kPaddedJetCount] = {};
    double higgs_e[kHiggsCount] = {};
    double higgs_px[kHiggsCount] = {};
    double higgs_py[kHiggsCount] = {};
    double higgs_pz[kHiggsCount] = {};
    double higgs_mass[kHiggsCount] = {};
    double higgs_pt[kHiggsCount] = {};
    double higgs_y[kHiggsCount] = {};
    int higgs_type[kHiggsCount] = {};
    int higgs_constituent1[kHiggsCount] = {};
    int higgs_constituent2[kHiggsCount] = {};
    int higgs_constituent1_source[kHiggsCount] = {};
    int higgs_constituent2_source[kHiggsCount] = {};
    double pair_mass[kPairCount] = {};
    double pair_dr[kPairCount] = {};
    double pair_dy[kPairCount] = {};
    double pair_dphi[kPairCount] = {};
    double m4h = 0.0;
    double pt4h = 0.0;
    double y4h = 0.0;
    double ht = 0.0;
    double event_centrality = 0.0;
    double event_sphericity = 0.0;

    tree.Branch("event_index", &event_index, "event_index/L");
    tree.Branch("weight", &weight, "weight/D");
    tree.Branch("raw_bjets", &raw_bjets, "raw_bjets/I");
    tree.Branch("accepted_bjets", &accepted_bjets, "accepted_bjets/I");
    tree.Branch("accepted_single_bjets", &accepted_single_bjets, "accepted_single_bjets/I");
    tree.Branch("accepted_merged_bjets", &accepted_merged_bjets, "accepted_merged_bjets/I");
    tree.Branch("accepted_cjet_candidates", &accepted_cjet_candidates, "accepted_cjet_candidates/I");
    tree.Branch("accepted_lightjet_candidates", &accepted_lightjet_candidates, "accepted_lightjet_candidates/I");
    tree.Branch("accepted_tag_equivalents", &accepted_tag_equivalents, "accepted_tag_equivalents/I");
    tree.Branch("reco_jets_considered", &reco_jets_considered, "reco_jets_considered/I");
    tree.Branch("reco_jets_used", &reco_jets_used, "reco_jets_used/I");
    tree.Branch("n_configurations", &n_configurations, "n_configurations/L");
    tree.Branch("n_true_single", &n_true_single, "n_true_single/I");
    tree.Branch("n_double_b", &n_double_b, "n_double_b/I");
    tree.Branch("n_c_mistag", &n_c_mistag, "n_c_mistag/I");
    tree.Branch("n_light_mistag", &n_light_mistag, "n_light_mistag/I");
    tree.Branch("n_merged", &n_merged, "n_merged/I");
    tree.Branch("category", &category, "category/I");
    tree.Branch("best_score", &best_score, "best_score/D");
    tree.Branch("second_score", &second_score, "second_score/D");
    tree.Branch("score_gap", &score_gap, "score_gap/D");
    tree.Branch("jet_pt", jet_pt, "jet_pt[8]/D");
    tree.Branch("higgs_e", higgs_e, "higgs_e[4]/D");
    tree.Branch("higgs_px", higgs_px, "higgs_px[4]/D");
    tree.Branch("higgs_py", higgs_py, "higgs_py[4]/D");
    tree.Branch("higgs_pz", higgs_pz, "higgs_pz[4]/D");
    tree.Branch("higgs_mass", higgs_mass, "higgs_mass[4]/D");
    tree.Branch("higgs_pt", higgs_pt, "higgs_pt[4]/D");
    tree.Branch("higgs_y", higgs_y, "higgs_y[4]/D");
    tree.Branch("higgs_type", higgs_type, "higgs_type[4]/I");
    tree.Branch("higgs_constituent1", higgs_constituent1, "higgs_constituent1[4]/I");
    tree.Branch("higgs_constituent2", higgs_constituent2, "higgs_constituent2[4]/I");
    tree.Branch("higgs_constituent1_source", higgs_constituent1_source,
                "higgs_constituent1_source[4]/I");
    tree.Branch("higgs_constituent2_source", higgs_constituent2_source,
                "higgs_constituent2_source[4]/I");
    tree.Branch("pair_mass", pair_mass, "pair_mass[6]/D");
    tree.Branch("pair_dr", pair_dr, "pair_dr[6]/D");
    tree.Branch("pair_dy", pair_dy, "pair_dy[6]/D");
    tree.Branch("pair_dphi", pair_dphi, "pair_dphi[6]/D");
    tree.Branch("m4h", &m4h, "m4h/D");
    tree.Branch("pt4h", &pt4h, "pt4h/D");
    tree.Branch("y4h", &y4h, "y4h/D");
    tree.Branch("ht", &ht, "ht/D");
    tree.Branch("centrality", &event_centrality, "centrality/D");
    tree.Branch("sphericity", &event_sphericity, "sphericity/D");

    Counter input_counter;
    Counter reconstructable_counter;
    std::array<Counter, 5> nmerged_counters = {};
    std::array<Counter, 3> category_counters = {};
    Long64_t invalid_multiplicity_entries = 0;
    Long64_t bjet_array_overflow_events = 0;
    Long64_t jet_array_overflow_events = 0;
    Long64_t failed_mistag_population_events = 0;
    Long64_t failed_reconstruction_events = 0;
    Long64_t true_b_upward_pt_migrations = 0;
    Long64_t true_b_downward_pt_migrations = 0;
    Long64_t non_b_upward_pt_migrations = 0;
    Long64_t non_b_downward_pt_migrations = 0;
    std::array<Long64_t, 3> true_b_upward_pt_migrations_by_raw_pt = {};
    std::array<Long64_t, 3> non_b_upward_pt_migrations_by_raw_pt = {};
    double max_smearing_mass_scaling_residual = 0.0;
    TRandom3 random(options.seed);

    for (event_index = 0; event_index < events_to_run; ++event_index) {
      chain.GetEntry(event_index);
      input_counter.fill(event_weight);

      if (number_bjets > kMaxInputJets) ++bjet_array_overflow_events;
      const int safe_number_bjets = std::max(0, std::min(number_bjets, kMaxInputJets));
      raw_bjets = safe_number_bjets;

      std::vector<Jet> true_bjets;
      true_bjets.reserve(static_cast<std::size_t>(safe_number_bjets));
      for (int index = 0; index < safe_number_bjets; ++index) {
        TLorentzVector raw;
        raw.SetPxPyPzE(bjets[1][index], bjets[2][index], bjets[3][index], bjets[0][index]);
        if (!std::isfinite(raw.Eta()) || std::fabs(raw.Eta()) >= kBJetEtaCut) continue;

        double mass_scaling_residual = 0.0;
        Jet jet;
        // Smear every eta-accepted stored jet exactly once before applying the
        // analysis-level pT threshold, including jets that migrate below it.
        jet.p4 = smearJetCMSUniformFourVector(
            raw, random, options.smear, mass_scaling_residual);
        max_smearing_mass_scaling_residual =
            std::max(max_smearing_mass_scaling_residual, mass_scaling_residual);
        const double raw_pt = raw.Pt();
        const bool raw_passes_pt = raw_pt > kBJetPtCut;
        const bool smeared_passes_pt = jet.p4.Pt() > kBJetPtCut;
        if (!raw_passes_pt && smeared_passes_pt) {
          ++true_b_upward_pt_migrations;
          const int raw_pt_bin = upwardMigrationRawPtBin(raw_pt);
          if (raw_pt_bin >= 0) {
            ++true_b_upward_pt_migrations_by_raw_pt[static_cast<std::size_t>(raw_pt_bin)];
          }
        }
        if (raw_passes_pt && !smeared_passes_pt) ++true_b_downward_pt_migrations;
        if (!smeared_passes_pt) continue;

        const int stored_multiplicity = b_hadron_multiplicity[index];
        if (stored_multiplicity < 1) {
          ++invalid_multiplicity_entries;
          throw std::runtime_error(
              "invalid bHadronMultiplicity < 1 at event " + std::to_string(event_index) +
              ", accepted true-b jet " + std::to_string(index) +
              "; refusing to infer a fallback multiplicity");
        }
        const int capped_multiplicity = std::min(2, stored_multiplicity);
        jet.tag_multiplicity = capped_multiplicity;
        jet.source = kTrueB;
        jet.original_index = index;
        true_bjets.push_back(jet);
      }
      std::sort(true_bjets.begin(), true_bjets.end(),
                [](const Jet& first, const Jet& second) { return first.p4.Pt() > second.p4.Pt(); });

      accepted_bjets = static_cast<int>(true_bjets.size());
      accepted_single_bjets = 0;
      accepted_merged_bjets = 0;
      for (const Jet& jet : true_bjets) {
        if (jet.tag_multiplicity >= 2) {
          ++accepted_merged_bjets;
        } else {
          ++accepted_single_bjets;
        }
      }

      std::vector<Jet> c_candidates;
      std::vector<Jet> light_candidates;
      if (need_mistag_branches) {
        if (number_jets > kMaxInputJets) ++jet_array_overflow_events;
        const int safe_number_jets = std::max(0, std::min(number_jets, kMaxInputJets));
        for (int index = 0; index < safe_number_jets; ++index) {
          TLorentzVector raw;
          raw.SetPxPyPzE(jets[1][index], jets[2][index], jets[3][index], jets[0][index]);
          if (!std::isfinite(raw.Eta()) || std::fabs(raw.Eta()) >= kBJetEtaCut) continue;

          double mass_scaling_residual = 0.0;
          Jet jet;
          // As for true-b jets, consume one Gaussian draw before the smeared
          // pT requirement so upward and downward threshold migrations enter.
          jet.p4 = smearJetCMSUniformFourVector(
              raw, random, options.smear, mass_scaling_residual);
          max_smearing_mass_scaling_residual =
              std::max(max_smearing_mass_scaling_residual, mass_scaling_residual);
          const double raw_pt = raw.Pt();
          const bool raw_passes_pt = raw_pt > kBJetPtCut;
          const bool smeared_passes_pt = jet.p4.Pt() > kBJetPtCut;
          if (!raw_passes_pt && smeared_passes_pt) {
            ++non_b_upward_pt_migrations;
            const int raw_pt_bin = upwardMigrationRawPtBin(raw_pt);
            if (raw_pt_bin >= 0) {
              ++non_b_upward_pt_migrations_by_raw_pt[static_cast<std::size_t>(raw_pt_bin)];
            }
          }
          if (raw_passes_pt && !smeared_passes_pt) ++non_b_downward_pt_migrations;
          if (!smeared_passes_pt || overlaps(jet.p4, true_bjets)) continue;
          jet.tag_multiplicity = 1;
          jet.source = charm_tag[index] > 0.0 ? kCharmMistag : kLightMistag;
          jet.original_index = index;
          if (jet.source == kCharmMistag) {
            c_candidates.push_back(jet);
          } else {
            light_candidates.push_back(jet);
          }
        }
        const auto by_pt = [](const Jet& first, const Jet& second) {
          return first.p4.Pt() > second.p4.Pt();
        };
        std::sort(c_candidates.begin(), c_candidates.end(), by_pt);
        std::sort(light_candidates.begin(), light_candidates.end(), by_pt);
      }

      accepted_cjet_candidates = static_cast<int>(c_candidates.size());
      accepted_lightjet_candidates = static_cast<int>(light_candidates.size());
      if (accepted_cjet_candidates < options.c_mistags ||
          accepted_lightjet_candidates < options.light_mistags) {
        ++failed_mistag_population_events;
        continue;
      }

      // Match the existing c3,d4 candidate convention: only the leading true-b
      // population needed by this process composition may enter pairing.  This
      // prevents backgrounds with extra real b jets (notably ttbar+4b) from
      // choosing a mass-optimized subset without paying the corresponding tag
      // probability for the extra candidates.
      const int required_true_tag_equivalents =
          8 - options.c_mistags - options.light_mistags;
      const int true_limit = std::min(
          std::min(static_cast<int>(true_bjets.size()), options.max_reco_jets),
          required_true_tag_equivalents);
      std::vector<Jet> candidate_jets(true_bjets.begin(), true_bjets.begin() + true_limit);
      std::vector<int> fixed_mistag_indices;
      for (int index = 0; index < options.c_mistags; ++index) {
        fixed_mistag_indices.push_back(static_cast<int>(candidate_jets.size()));
        candidate_jets.push_back(c_candidates[index]);
      }
      for (int index = 0; index < options.light_mistags; ++index) {
        fixed_mistag_indices.push_back(static_cast<int>(candidate_jets.size()));
        candidate_jets.push_back(light_candidates[index]);
      }

      reco_jets_considered = static_cast<int>(candidate_jets.size());
      accepted_tag_equivalents = options.c_mistags + options.light_mistags;
      for (int index = 0; index < true_limit; ++index) {
        accepted_tag_equivalents += candidate_jets[index].tag_multiplicity;
      }

      BestReconstruction reconstruction =
          reconstruct(candidate_jets, fixed_mistag_indices,
                      options.c_mistags, options.light_mistags);
      if (!reconstruction.valid) {
        ++failed_reconstruction_events;
        continue;
      }
      std::sort(reconstruction.candidates.begin(), reconstruction.candidates.end(),
                [](const Candidate& first, const Candidate& second) {
                  return first.p4.Pt() > second.p4.Pt();
                });

      n_merged = 0;
      std::vector<int> used_indices;
      for (const Candidate& candidate : reconstruction.candidates) {
        n_merged += candidate.type;
        used_indices.push_back(candidate.first_jet);
        if (candidate.second_jet >= 0) used_indices.push_back(candidate.second_jet);
      }
      std::sort(used_indices.begin(), used_indices.end());
      used_indices.erase(std::unique(used_indices.begin(), used_indices.end()), used_indices.end());

      n_double_b = n_merged;
      n_true_single = 8 - 2 * n_double_b - options.c_mistags - options.light_mistags;
      reco_jets_used = static_cast<int>(used_indices.size());
      if (n_true_single < 0 ||
          2 * n_double_b + n_true_single + n_c_mistag + n_light_mistag != 8 ||
          reco_jets_used != 8 - n_double_b) {
        throw std::runtime_error("internal tag-composition closure failure");
      }
      category = n_merged == 0 ? 0 : (n_merged <= 2 ? 1 : 2);
      n_configurations = reconstruction.configurations;
      best_score = reconstruction.best_score;
      second_score = std::isfinite(reconstruction.second_score) ? reconstruction.second_score : -1.0;
      score_gap = second_score >= 0.0 ? second_score - best_score : -1.0;
      weight = event_weight;

      std::fill(std::begin(jet_pt), std::end(jet_pt), 0.0);
      std::vector<Jet> selected_jets;
      selected_jets.reserve(used_indices.size());
      for (int index : used_indices) selected_jets.push_back(candidate_jets[index]);
      std::sort(selected_jets.begin(), selected_jets.end(),
                [](const Jet& first, const Jet& second) { return first.p4.Pt() > second.p4.Pt(); });
      for (std::size_t index = 0;
           index < selected_jets.size() && index < static_cast<std::size_t>(kPaddedJetCount); ++index) {
        jet_pt[index] = selected_jets[index].p4.Pt();
      }

      TLorentzVector four_higgs;
      for (int index = 0; index < kHiggsCount; ++index) {
        const Candidate& candidate = reconstruction.candidates[index];
        higgs_e[index] = candidate.p4.E();
        higgs_px[index] = candidate.p4.Px();
        higgs_py[index] = candidate.p4.Py();
        higgs_pz[index] = candidate.p4.Pz();
        higgs_mass[index] = candidate.p4.M();
        higgs_pt[index] = candidate.p4.Pt();
        higgs_y[index] = candidate.p4.Rapidity();
        higgs_type[index] = candidate.type;
        higgs_constituent1[index] = candidate.first_jet;
        higgs_constituent2[index] = candidate.second_jet;
        higgs_constituent1_source[index] = candidate_jets[candidate.first_jet].source;
        higgs_constituent2_source[index] =
            candidate.second_jet >= 0 ? candidate_jets[candidate.second_jet].source : -1;
        four_higgs += candidate.p4;
      }

      int pair_index = 0;
      for (int first = 0; first < kHiggsCount; ++first) {
        for (int second = first + 1; second < kHiggsCount; ++second) {
          const TLorentzVector pair = reconstruction.candidates[first].p4 +
                                      reconstruction.candidates[second].p4;
          pair_mass[pair_index] = pair.M();
          pair_dr[pair_index] = deltaR(reconstruction.candidates[first].p4,
                                       reconstruction.candidates[second].p4);
          pair_dy[pair_index] = std::fabs(reconstruction.candidates[first].p4.Rapidity() -
                                          reconstruction.candidates[second].p4.Rapidity());
          pair_dphi[pair_index] = absoluteDeltaPhi(reconstruction.candidates[first].p4.Phi(),
                                                   reconstruction.candidates[second].p4.Phi());
          ++pair_index;
        }
      }

      m4h = four_higgs.M();
      pt4h = four_higgs.Pt();
      y4h = four_higgs.Rapidity();
      ht = 0.0;
      for (const Jet& jet : selected_jets) ht += jet.p4.Pt();
      event_centrality = centrality(selected_jets);
      event_sphericity = transverseSphericity(selected_jets);

      reconstructable_counter.fill(event_weight);
      nmerged_counters[n_merged].fill(event_weight);
      category_counters[category].fill(event_weight);
      tree.Fill();
    }

    output.cd();
    tree.Write();
    TNamed observable_schema("observable_schema", "resonance-hybrid-v1");
    observable_schema.Write();
    TNamed method_version("method_version", kMethodVersion);
    method_version.Write();
    TNamed preprocessing_version("preprocessing_version", kPreprocessingVersion);
    preprocessing_version.Write();
    TNamed branch_schema("branch_schema_json", branchSchemaJson());
    branch_schema.Write();
    const std::vector<std::string> feature_names = featureNames();
    TNamed feature_names_metadata("feature_names_json", jsonStringArray(feature_names).c_str());
    feature_names_metadata.Write();
    TNamed audit_branch_names(
        "audit_branch_names_json",
        "[\"raw_bjets\",\"accepted_bjets\",\"accepted_single_bjets\","
        "\"accepted_merged_bjets\",\"accepted_cjet_candidates\","
        "\"accepted_lightjet_candidates\",\"accepted_tag_equivalents\","
        "\"reco_jets_considered\",\"reco_jets_used\",\"n_configurations\","
        "\"n_true_single\",\"n_double_b\",\"n_c_mistag\","
        "\"n_light_mistag\",\"higgs_constituent1_source\","
        "\"higgs_constituent2_source\"]");
    audit_branch_names.Write();
    TNamed category_definition(
        "category_definition_json",
        "{\"0\":\"resolved (n_merged=0)\",\"1\":\"mixed (n_merged=1 or 2)\","
        "\"2\":\"boosted (n_merged=3 or 4)\"}");
    category_definition.Write();
    TNamed pair_order("pair_order_json", "[\"12\",\"13\",\"14\",\"23\",\"24\",\"34\"]");
    pair_order.Write();
    std::ostringstream smearing_description;
    smearing_description
        << "{\"enabled\":" << (options.smear ? "true" : "false")
        << ",\"seed\":" << options.seed
        << ",\"preprocessing_version\":\"" << kPreprocessingVersion << "\""
        << ",\"model_id\":\"" << kSmearingModelId << "\""
        << ",\"fourvector_scaling\":\"uniform_correlated\""
        << ",\"correlated_mass_scaling\":true"
        << ",\"preserves_jet_mass\":false"
        << ",\"gaussian_draws_per_jet\":1"
        << ",\"energy_floor_gev\":1e-6"
        << ",\"eta_preselection\":\"finite |eta|<2.5 before smearing\""
        << ",\"pt_threshold\":\"smeared pT>20 GeV\""
        << ",\"smear_before_pt_threshold\":true"
        << ",\"acceptance_order\":\"raw_abs_eta_then_smear_then_smeared_pt\""
        << ",\"model\":\"CMS energy resolution with p'^mu=(E'/E)p^mu and correlated mass scaling\"}";
    TNamed smearing_metadata("smearing_json", smearing_description.str().c_str());
    smearing_metadata.Write();
    TNamed tagging_metadata(
        "tagging_definition_json",
        "{\"efficiencies_applied\":false,\"weight\":\"raw evweight\","
        "\"closure\":\"2*n_double_b+n_true_single+n_c_mistag+n_light_mistag=8\"}");
    tagging_metadata.Write();
    TParameter<int>("feature_count", static_cast<int>(feature_names.size())).Write();
    TParameter<int>("max_reco_true_bjets", options.max_reco_jets).Write();
    TParameter<int>("c_mistags", options.c_mistags).Write();
    TParameter<int>("light_mistags", options.light_mistags).Write();
    TParameter<int>("smearing_enabled", options.smear ? 1 : 0).Write();
    TParameter<Long64_t>("events_processed", input_counter.events).Write();
    TParameter<Long64_t>("events_reconstructable", reconstructable_counter.events).Write();
    TParameter<double>("total_weight_in", input_counter.sumw).Write();
    TParameter<double>("total_weight_out", reconstructable_counter.sumw).Write();
    output.Close();

    std::ofstream summary(options.output_json.c_str());
    if (!summary) {
      throw std::runtime_error("cannot create summary JSON " + options.output_json);
    }
    summary << std::setprecision(17);
    summary << "{\n"
            << "  \"schema\": \"resonance-hybrid-v1\",\n"
            << "  \"method_version\": \"" << kMethodVersion << "\",\n"
            << "  \"preprocessing_version\": \"" << kPreprocessingVersion << "\",\n"
            << "  \"input\": \"" << jsonEscape(options.input) << "\",\n"
            << "  \"output_root\": \"" << jsonEscape(options.output_root) << "\",\n"
            << "  \"events_available\": " << available_events << ",\n"
            << "  \"events_requested\": " << events_to_run << ",\n"
            << "  \"pt_cut_gev\": " << kBJetPtCut << ",\n"
            << "  \"eta_cut\": " << kBJetEtaCut << ",\n"
            << "  \"max_reco_true_bjets\": " << options.max_reco_jets << ",\n"
            << "  \"c_mistags\": " << options.c_mistags << ",\n"
            << "  \"light_mistags\": " << options.light_mistags << ",\n"
            << "  \"tag_efficiencies_applied\": false,\n"
            << "  \"weight_definition\": \"raw evweight\",\n"
            << "  \"smearing\": {\"enabled\": " << (options.smear ? "true" : "false")
            << ", \"seed\": " << options.seed
            << ", \"preprocessing_version\": \"" << kPreprocessingVersion << "\""
            << ", \"model_id\": \"" << kSmearingModelId << "\""
            << ", \"fourvector_scaling\": \"uniform_correlated\""
            << ", \"correlated_mass_scaling\": true"
            << ", \"preserves_jet_mass\": false"
            << ", \"gaussian_draws_per_jet\": 1"
            << ", \"energy_floor_gev\": 1e-6"
            << ", \"eta_preselection\": \"finite |eta|<2.5 before smearing\""
            << ", \"pt_threshold\": \"smeared pT>20 GeV\""
            << ", \"smear_before_pt_threshold\": true"
            << ", \"acceptance_order\": \"raw_abs_eta_then_smear_then_smeared_pt\"},\n"
            << "  \"input_counter\": ";
    writeCounter(summary, input_counter, 0);
    summary << ",\n  \"reconstructable_counter\": ";
    writeCounter(summary, reconstructable_counter, 0);
    summary << ",\n"
            << "  \"count_efficiency\": "
            << fraction(reconstructable_counter.events, input_counter.events) << ",\n"
            << "  \"weighted_efficiency\": "
            << fraction(reconstructable_counter.sumw, input_counter.sumw) << ",\n"
            << "  \"n_merged\": {\n";
    for (int index = 0; index < 5; ++index) {
      summary << "    \"" << index << "\": ";
      writeCounter(summary, nmerged_counters[index], 0);
      summary << (index == 4 ? "\n" : ",\n");
    }
    summary << "  },\n  \"categories\": {\n";
    const std::array<std::string, 3> category_names = {"resolved", "mixed", "boosted"};
    for (int index = 0; index < 3; ++index) {
      summary << "    \"" << category_names[index] << "\": ";
      writeCounter(summary, category_counters[index], 0);
      summary << (index == 2 ? "\n" : ",\n");
    }
    summary << "  },\n"
            << "  \"diagnostics\": {\n"
            << "    \"invalid_multiplicity_entries\": " << invalid_multiplicity_entries << ",\n"
            << "    \"bjet_array_overflow_events\": " << bjet_array_overflow_events << ",\n"
            << "    \"jet_array_overflow_events\": " << jet_array_overflow_events << ",\n"
            << "    \"failed_mistag_population_events\": " << failed_mistag_population_events << ",\n"
            << "    \"failed_reconstruction_events\": " << failed_reconstruction_events << ",\n"
            << "    \"true_b_upward_pt_migrations\": " << true_b_upward_pt_migrations << ",\n"
            << "    \"true_b_downward_pt_migrations\": " << true_b_downward_pt_migrations << ",\n"
            << "    \"non_b_upward_pt_migrations\": " << non_b_upward_pt_migrations << ",\n"
            << "    \"non_b_downward_pt_migrations\": " << non_b_downward_pt_migrations << ",\n"
            << "    \"true_b_upward_pt_migrations_by_raw_pt_gev\": "
            << "{\"[10,12)\": " << true_b_upward_pt_migrations_by_raw_pt[0]
            << ", \"[12,15)\": " << true_b_upward_pt_migrations_by_raw_pt[1]
            << ", \"[15,20]\": " << true_b_upward_pt_migrations_by_raw_pt[2] << "},\n"
            << "    \"non_b_upward_pt_migrations_by_raw_pt_gev\": "
            << "{\"[10,12)\": " << non_b_upward_pt_migrations_by_raw_pt[0]
            << ", \"[12,15)\": " << non_b_upward_pt_migrations_by_raw_pt[1]
            << ", \"[15,20]\": " << non_b_upward_pt_migrations_by_raw_pt[2] << "},\n"
            << "    \"max_smearing_mass_scaling_residual_gev\": "
            << max_smearing_mass_scaling_residual << "\n"
            << "  }\n"
            << "}\n";
    summary.close();

    std::cout << "Processed " << input_counter.events << " events; reconstructed "
              << reconstructable_counter.events << "\n"
              << "Input sumw: " << input_counter.sumw
              << "; reconstructed sumw: " << reconstructable_counter.sumw << "\n"
              << "ROOT output: " << options.output_root << "\n"
              << "JSON summary: " << options.output_json << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
