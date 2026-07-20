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
constexpr int kMaxInputFatJets = 100;
constexpr int kHiggsCount = 4;
constexpr int kPairCount = 6;
constexpr int kPaddedJetCount = 8;
constexpr int kPaddedFatJetCount = 4;
constexpr int kDefaultMaxRecoJets = 10;
constexpr int kHardMaxRecoJets = 10;
constexpr double kBJetPtCut = 20.0;
constexpr double kBJetEtaCut = 2.5;
constexpr double kDuplicateJetDeltaR = 0.05;
constexpr double kFatJetPtCut = 300.0;
constexpr double kFatJetEtaCut = 2.5;
constexpr double kFatJetOverlapDeltaR = 0.8;
constexpr double kHiggsMass = 125.0;
constexpr const char* kMethodVersion = "fatjet-ak8-softdrop-v1";
constexpr const char* kPreprocessingVersion = "fatjet-ak8-preprocessing-v1";
constexpr const char* kSmearingModelId = "cms-energy-uniform-fourvector-v1";
constexpr double kEpsBbNominal = 0.7225;
constexpr double kFakeBbNominal = 0.10;
constexpr double kEpsBbConservative = 0.30;
constexpr double kFakeBbConservative = 0.01;

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
  int source = kTrueB;
  int original_index = -1;
};

struct FatJet {
  TLorentzVector p4;
  TLorentzVector softdrop_p4;
  double tau21 = 0.0;
  int b_hadrons = 0;
  int c_hadrons = 0;
  int original_index = -1;

  bool trueDoubleB() const { return b_hadrons == 2 || b_hadrons == 3; }
};

struct Candidate {
  TLorentzVector p4;
  double tag_mass = 0.0;
  int type = 0;  // 0: resolved AK4 pair, 1: passing AK8 candidate.
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
      const double residual = (candidate.tag_mass - kHiggsMass) / kHiggsMass;
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

  const std::array<const char*, 10> required = {
      "evweight", "thebJets", "numbJets", "bHadronMultiplicity",
      "numFatJets", "theFatJets", "theSoftDropFatJets", "tau21FatJets",
      "bHadronMultiplicityFatJets", "cHadronMultiplicityFatJets"};
  for (const char* branch : required) {
    if (tree->GetBranch(branch) == nullptr) {
      if (std::string(branch).find("FatJet") != std::string::npos ||
          std::string(branch) == "numFatJets") {
        throw std::runtime_error(
            "Data tree in " + path +
            " lacks the AK8 branches; regenerate it with HwSim FatJets Yes");
      }
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

BestReconstruction reconstruct(const std::vector<Jet>& resolved_jets,
                               const std::vector<int>& passing_fat_indices,
                               const std::vector<FatJet>& fat_jets) {
  BestReconstruction best;
  const int n_resolved = kHiggsCount - static_cast<int>(passing_fat_indices.size());
  if (n_resolved < 0 ||
      static_cast<int>(resolved_jets.size()) != 2 * n_resolved) {
    return best;
  }

  std::vector<int> resolved_indices(resolved_jets.size());
  for (std::size_t index = 0; index < resolved_indices.size(); ++index) {
    resolved_indices[index] = static_cast<int>(index);
  }
  pairings(resolved_indices, [&](const PairList& resolved_pairs) {
    std::array<Candidate, kHiggsCount> proposal = {};
    int candidate_index = 0;
    for (int fat_index : passing_fat_indices) {
      Candidate& candidate = proposal[candidate_index++];
      candidate.p4 = fat_jets[static_cast<std::size_t>(fat_index)].p4;
      candidate.tag_mass =
          fat_jets[static_cast<std::size_t>(fat_index)].softdrop_p4.M();
      candidate.type = 1;
      candidate.first_jet = fat_index;
      candidate.second_jet = -1;
    }
    for (const std::array<int, 2>& pair : resolved_pairs) {
      Candidate& candidate = proposal[candidate_index++];
      candidate.p4 = resolved_jets[static_cast<std::size_t>(pair[0])].p4 +
                     resolved_jets[static_cast<std::size_t>(pair[1])].p4;
      candidate.tag_mass = candidate.p4.M();
      candidate.type = 0;
      candidate.first_jet = pair[0];
      candidate.second_jet = pair[1];
    }
    if (candidate_index == kHiggsCount) best.consider(proposal);
  });
  return best;
}

bool overlapsFat(const TLorentzVector& jet,
                 const std::vector<int>& passing_fat_indices,
                 const std::vector<FatJet>& fat_jets) {
  for (int index : passing_fat_indices) {
    if (deltaR(jet, fat_jets[static_cast<std::size_t>(index)].p4) <
        kFatJetOverlapDeltaR) {
      return true;
    }
  }
  return false;
}

double tagPatternProbability(const std::vector<FatJet>& fat_jets,
                             int pattern,
                             double eps_bb,
                             double fake_bb) {
  double probability = 1.0;
  for (std::size_t index = 0; index < fat_jets.size(); ++index) {
    const bool pass = (pattern & (1 << index)) != 0;
    const double pass_probability =
        fat_jets[index].trueDoubleB() ? eps_bb : fake_bb;
    probability *= pass ? pass_probability : (1.0 - pass_probability);
  }
  return probability;
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
      "n_ak8_retained", "n_merged", "category", "best_score",
      "second_score", "score_gap"};
  for (int index = 1; index <= kPaddedJetCount; ++index) {
    names.push_back("jet_pt_" + std::to_string(index));
  }
  const std::array<std::string, 9> candidate_fields = {
      "e", "px", "py", "pz", "mass", "tag_mass", "pt", "y", "type"};
  for (const std::string& field : candidate_fields) {
    for (int index = 1; index <= kHiggsCount; ++index) {
      names.push_back("higgs_" + field + "_" + std::to_string(index));
    }
  }
  const std::array<std::string, 5> fat_fields = {
      "pt", "eta", "mass", "softdrop_mass", "tau21"};
  for (const std::string& field : fat_fields) {
    for (int index = 1; index <= kPaddedFatJetCount; ++index) {
      names.push_back("fat_" + field + "_" + std::to_string(index));
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
"schema":"fatjet-ak8-softdrop-v1",
"candidate_order":"descending_pt",
"hypotheses":"one row per reconstructable AK8 pass/fail pattern; raw generator weight is repeated",
"branches":{
"event_index":"Long64_t","hypothesis_index":"Int_t AK8 pass bitmask","weight":"Double_t raw generator weight; no tag efficiency applied",
"n_ak8_eligible":"Int_t before leading-four truncation","n_ak8_retained":"Int_t [0,4] taggable candidate multiplicity",
"n_true_single":"Int_t","n_c_mistag":"Int_t","n_light_mistag":"Int_t",
"n_true_fat_pass":"Int_t","n_true_fat_fail":"Int_t","n_fake_fat_pass":"Int_t","n_fake_fat_fail":"Int_t",
"n_merged":"Int_t [0,4]","category":"Int_t 0=resolved,1=mixed,2=boosted",
"best_score":"Double_t","second_score":"Double_t; -1 if unavailable","score_gap":"Double_t; -1 if unavailable",
"jet_pt":"Double_t[8]","higgs_e":"Double_t[4]","higgs_px":"Double_t[4]","higgs_py":"Double_t[4]","higgs_pz":"Double_t[4]","higgs_mass":"Double_t[4]","higgs_tag_mass":"Double_t[4]","higgs_pt":"Double_t[4]","higgs_y":"Double_t[4]","higgs_type":"Int_t[4] 0=resolved,1=AK8",
"fat_pt":"Double_t[4]","fat_eta":"Double_t[4]","fat_mass":"Double_t[4]","fat_softdrop_mass":"Double_t[4]","fat_tau21":"Double_t[4]",
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

}  // namespace


namespace {

TLorentzVector scaledFourVector(const TLorentzVector& input, double scale) {
  TLorentzVector output;
  output.SetPxPyPzE(scale * input.Px(), scale * input.Py(),
                   scale * input.Pz(), scale * input.E());
  return output;
}

template <typename T, std::size_t N>
void fillArray(T (&values)[N], const T& value) {
  std::fill(std::begin(values), std::end(values), value);
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
    const Long64_t events_to_run = options.max_events > 0
        ? std::min(options.max_events, available_events) : available_events;

    double event_weight = 0.0;
    int number_bjets = 0;
    int number_jets = 0;
    int number_fatjets = 0;
    double bjets[5][kMaxInputJets] = {};
    double jets[5][kMaxInputJets] = {};
    double charm_tag[kMaxInputJets] = {};
    int b_hadron_multiplicity[kMaxInputJets] = {};
    double fatjets[4][kMaxInputFatJets] = {};
    double softdrop_fatjets[4][kMaxInputFatJets] = {};
    double fat_tau21[kMaxInputFatJets] = {};
    int fat_b_hadrons[kMaxInputFatJets] = {};
    int fat_c_hadrons[kMaxInputFatJets] = {};

    chain.SetBranchAddress("evweight", &event_weight);
    chain.SetBranchAddress("thebJets", &bjets);
    chain.SetBranchAddress("numbJets", &number_bjets);
    chain.SetBranchAddress("bHadronMultiplicity", &b_hadron_multiplicity);
    chain.SetBranchAddress("numFatJets", &number_fatjets);
    chain.SetBranchAddress("theFatJets", &fatjets);
    chain.SetBranchAddress("theSoftDropFatJets", &softdrop_fatjets);
    chain.SetBranchAddress("tau21FatJets", &fat_tau21);
    chain.SetBranchAddress("bHadronMultiplicityFatJets", &fat_b_hadrons);
    chain.SetBranchAddress("cHadronMultiplicityFatJets", &fat_c_hadrons);
    if (need_mistag_branches) {
      chain.SetBranchAddress("theJets", &jets);
      chain.SetBranchAddress("numJets", &number_jets);
      chain.SetBranchAddress("cTag", &charm_tag);
    }

    TFile output(options.output_root.c_str(), "RECREATE");
    if (output.IsZombie()) {
      throw std::runtime_error("cannot create output ROOT file " + options.output_root);
    }
    TTree tree("ResonanceFeatures", "AK8-aware four-Higgs hypothesis feature tree");

    Long64_t event_index = -1;
    int hypothesis_index = -1;
    double weight = 0.0;
    int raw_bjets = 0;
    int accepted_bjets = 0;
    int accepted_cjet_candidates = 0;
    int accepted_lightjet_candidates = 0;
    int n_ak8_eligible = 0;
    int n_ak8_retained = 0;
    int n_ak8_hh_diagnostic = 0;
    int n_true_single = 0;
    int n_c_mistag = 0;
    int n_light_mistag = 0;
    int n_true_fat_pass = 0;
    int n_true_fat_fail = 0;
    int n_fake_fat_pass = 0;
    int n_fake_fat_fail = 0;
    int n_merged = 0;
    int category = -1;
    int reco_jets_used = 0;
    Long64_t n_configurations = 0;
    double best_score = 0.0;
    double second_score = -1.0;
    double score_gap = -1.0;
    double jet_pt[kPaddedJetCount] = {};
    double higgs_e[kHiggsCount] = {};
    double higgs_px[kHiggsCount] = {};
    double higgs_py[kHiggsCount] = {};
    double higgs_pz[kHiggsCount] = {};
    double higgs_mass[kHiggsCount] = {};
    double higgs_tag_mass[kHiggsCount] = {};
    double higgs_pt[kHiggsCount] = {};
    double higgs_y[kHiggsCount] = {};
    int higgs_type[kHiggsCount] = {};
    int higgs_constituent1[kHiggsCount] = {};
    int higgs_constituent2[kHiggsCount] = {};
    int higgs_constituent1_source[kHiggsCount] = {};
    int higgs_constituent2_source[kHiggsCount] = {};
    double fat_pt[kPaddedFatJetCount] = {};
    double fat_eta[kPaddedFatJetCount] = {};
    double fat_mass[kPaddedFatJetCount] = {};
    double fat_softdrop_mass[kPaddedFatJetCount] = {};
    double fat_tau21_selected[kPaddedFatJetCount] = {};
    int fat_b_hadron_multiplicity[kPaddedFatJetCount] = {};
    int fat_c_hadron_multiplicity[kPaddedFatJetCount] = {};
    int fat_tag_kind[kPaddedFatJetCount] = {};
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
    tree.Branch("hypothesis_index", &hypothesis_index, "hypothesis_index/I");
    tree.Branch("weight", &weight, "weight/D");
    tree.Branch("raw_bjets", &raw_bjets, "raw_bjets/I");
    tree.Branch("accepted_bjets", &accepted_bjets, "accepted_bjets/I");
    tree.Branch("accepted_cjet_candidates", &accepted_cjet_candidates,
                "accepted_cjet_candidates/I");
    tree.Branch("accepted_lightjet_candidates", &accepted_lightjet_candidates,
                "accepted_lightjet_candidates/I");
    tree.Branch("n_ak8_eligible", &n_ak8_eligible, "n_ak8_eligible/I");
    tree.Branch("n_ak8_retained", &n_ak8_retained, "n_ak8_retained/I");
    tree.Branch("n_ak8_hh_diagnostic", &n_ak8_hh_diagnostic,
                "n_ak8_hh_diagnostic/I");
    tree.Branch("n_true_single", &n_true_single, "n_true_single/I");
    tree.Branch("n_c_mistag", &n_c_mistag, "n_c_mistag/I");
    tree.Branch("n_light_mistag", &n_light_mistag, "n_light_mistag/I");
    tree.Branch("n_true_fat_pass", &n_true_fat_pass, "n_true_fat_pass/I");
    tree.Branch("n_true_fat_fail", &n_true_fat_fail, "n_true_fat_fail/I");
    tree.Branch("n_fake_fat_pass", &n_fake_fat_pass, "n_fake_fat_pass/I");
    tree.Branch("n_fake_fat_fail", &n_fake_fat_fail, "n_fake_fat_fail/I");
    tree.Branch("n_merged", &n_merged, "n_merged/I");
    tree.Branch("category", &category, "category/I");
    tree.Branch("reco_jets_used", &reco_jets_used, "reco_jets_used/I");
    tree.Branch("n_configurations", &n_configurations, "n_configurations/L");
    tree.Branch("best_score", &best_score, "best_score/D");
    tree.Branch("second_score", &second_score, "second_score/D");
    tree.Branch("score_gap", &score_gap, "score_gap/D");
    tree.Branch("jet_pt", jet_pt, "jet_pt[8]/D");
    tree.Branch("higgs_e", higgs_e, "higgs_e[4]/D");
    tree.Branch("higgs_px", higgs_px, "higgs_px[4]/D");
    tree.Branch("higgs_py", higgs_py, "higgs_py[4]/D");
    tree.Branch("higgs_pz", higgs_pz, "higgs_pz[4]/D");
    tree.Branch("higgs_mass", higgs_mass, "higgs_mass[4]/D");
    tree.Branch("higgs_tag_mass", higgs_tag_mass, "higgs_tag_mass[4]/D");
    tree.Branch("higgs_pt", higgs_pt, "higgs_pt[4]/D");
    tree.Branch("higgs_y", higgs_y, "higgs_y[4]/D");
    tree.Branch("higgs_type", higgs_type, "higgs_type[4]/I");
    tree.Branch("higgs_constituent1", higgs_constituent1,
                "higgs_constituent1[4]/I");
    tree.Branch("higgs_constituent2", higgs_constituent2,
                "higgs_constituent2[4]/I");
    tree.Branch("higgs_constituent1_source", higgs_constituent1_source,
                "higgs_constituent1_source[4]/I");
    tree.Branch("higgs_constituent2_source", higgs_constituent2_source,
                "higgs_constituent2_source[4]/I");
    tree.Branch("fat_pt", fat_pt, "fat_pt[4]/D");
    tree.Branch("fat_eta", fat_eta, "fat_eta[4]/D");
    tree.Branch("fat_mass", fat_mass, "fat_mass[4]/D");
    tree.Branch("fat_softdrop_mass", fat_softdrop_mass,
                "fat_softdrop_mass[4]/D");
    tree.Branch("fat_tau21", fat_tau21_selected, "fat_tau21[4]/D");
    tree.Branch("fat_b_hadron_multiplicity", fat_b_hadron_multiplicity,
                "fat_b_hadron_multiplicity[4]/I");
    tree.Branch("fat_c_hadron_multiplicity", fat_c_hadron_multiplicity,
                "fat_c_hadron_multiplicity[4]/I");
    tree.Branch("fat_tag_kind", fat_tag_kind, "fat_tag_kind[4]/I");
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
    Counter events_with_reconstruction;
    Counter hypothesis_row_counter;
    std::array<Counter, 3> category_row_counters = {};
    Long64_t failed_patterns = 0;
    Long64_t fat_candidate_overflow_events = 0;
    Long64_t hh_diagnostic_events = 0;
    double max_smearing_mass_scaling_residual = 0.0;
    double max_pattern_probability_residual_nominal = 0.0;
    double max_pattern_probability_residual_conservative = 0.0;
    double selected_probability_sum_nominal = 0.0;
    double selected_probability_sum_conservative = 0.0;
    TRandom3 random(options.seed);

    for (event_index = 0; event_index < events_to_run; ++event_index) {
      chain.GetEntry(event_index);
      input_counter.fill(event_weight);
      if (number_fatjets < 0 || number_fatjets > kMaxInputFatJets) {
        throw std::runtime_error(
            "numFatJets is outside the fixed [0,100] branch range at event " +
            std::to_string(event_index));
      }

      const int safe_bjets = std::max(0, std::min(number_bjets, kMaxInputJets));
      raw_bjets = safe_bjets;
      std::vector<Jet> true_bjets;
      for (int index = 0; index < safe_bjets; ++index) {
        TLorentzVector raw;
        raw.SetPxPyPzE(bjets[1][index], bjets[2][index],
                       bjets[3][index], bjets[0][index]);
        if (!std::isfinite(raw.Eta()) || std::fabs(raw.Eta()) >= kBJetEtaCut) continue;
        double residual = 0.0;
        Jet jet;
        jet.p4 = smearJetCMSUniformFourVector(raw, random, options.smear, residual);
        max_smearing_mass_scaling_residual =
            std::max(max_smearing_mass_scaling_residual, residual);
        if (jet.p4.Pt() <= kBJetPtCut) continue;
        jet.source = kTrueB;
        jet.original_index = index;
        true_bjets.push_back(jet);
      }
      const auto by_pt = [](const Jet& first, const Jet& second) {
        return first.p4.Pt() > second.p4.Pt();
      };
      std::sort(true_bjets.begin(), true_bjets.end(), by_pt);
      accepted_bjets = static_cast<int>(true_bjets.size());

      std::vector<Jet> c_candidates;
      std::vector<Jet> light_candidates;
      if (need_mistag_branches) {
        const int safe_jets = std::max(0, std::min(number_jets, kMaxInputJets));
        for (int index = 0; index < safe_jets; ++index) {
          TLorentzVector raw;
          raw.SetPxPyPzE(jets[1][index], jets[2][index],
                         jets[3][index], jets[0][index]);
          if (!std::isfinite(raw.Eta()) || std::fabs(raw.Eta()) >= kBJetEtaCut) continue;
          double residual = 0.0;
          Jet jet;
          jet.p4 = smearJetCMSUniformFourVector(raw, random, options.smear, residual);
          max_smearing_mass_scaling_residual =
              std::max(max_smearing_mass_scaling_residual, residual);
          if (jet.p4.Pt() <= kBJetPtCut || overlaps(jet.p4, true_bjets)) continue;
          jet.source = charm_tag[index] > 0.0 ? kCharmMistag : kLightMistag;
          jet.original_index = index;
          (jet.source == kCharmMistag ? c_candidates : light_candidates).push_back(jet);
        }
        std::sort(c_candidates.begin(), c_candidates.end(), by_pt);
        std::sort(light_candidates.begin(), light_candidates.end(), by_pt);
      }
      accepted_cjet_candidates = static_cast<int>(c_candidates.size());
      accepted_lightjet_candidates = static_cast<int>(light_candidates.size());

      std::vector<FatJet> eligible_fatjets;
      n_ak8_hh_diagnostic = 0;
      for (int index = 0; index < number_fatjets; ++index) {
        TLorentzVector raw;
        raw.SetPxPyPzE(fatjets[1][index], fatjets[2][index],
                       fatjets[3][index], fatjets[0][index]);
        if (!std::isfinite(raw.Eta()) || std::fabs(raw.Eta()) >= kFatJetEtaCut) continue;
        TLorentzVector raw_softdrop;
        raw_softdrop.SetPxPyPzE(softdrop_fatjets[1][index],
                                softdrop_fatjets[2][index],
                                softdrop_fatjets[3][index],
                                softdrop_fatjets[0][index]);
        double residual = 0.0;
        FatJet fat;
        fat.p4 = smearJetCMSUniformFourVector(raw, random, options.smear, residual);
        max_smearing_mass_scaling_residual =
            std::max(max_smearing_mass_scaling_residual, residual);
        const double scale = raw.E() > 0.0 ? fat.p4.E() / raw.E() : 1.0;
        fat.softdrop_p4 = scaledFourVector(raw_softdrop, scale);
        fat.tau21 = fat_tau21[index];
        fat.b_hadrons = fat_b_hadrons[index];
        fat.c_hadrons = fat_c_hadrons[index];
        fat.original_index = index;
        if (fat.p4.Pt() <= kFatJetPtCut) continue;
        if (!std::isfinite(fat.tau21) || fat.b_hadrons < 0 || fat.c_hadrons < 0) {
          throw std::runtime_error(
              "invalid AK8 substructure/flavour value at event " +
              std::to_string(event_index));
        }
        if (fat.b_hadrons >= 4) {
          ++n_ak8_hh_diagnostic;
          continue;
        }
        eligible_fatjets.push_back(fat);
      }
      std::sort(eligible_fatjets.begin(), eligible_fatjets.end(),
                [](const FatJet& first, const FatJet& second) {
                  return first.p4.Pt() > second.p4.Pt();
                });
      n_ak8_eligible = static_cast<int>(eligible_fatjets.size());
      if (n_ak8_eligible > kPaddedFatJetCount) ++fat_candidate_overflow_events;
      if (n_ak8_hh_diagnostic > 0) ++hh_diagnostic_events;
      if (eligible_fatjets.size() > static_cast<std::size_t>(kPaddedFatJetCount)) {
        eligible_fatjets.resize(kPaddedFatJetCount);
      }
      n_ak8_retained = static_cast<int>(eligible_fatjets.size());

      const int pattern_count = 1 << n_ak8_retained;
      double probability_sum_nominal = 0.0;
      double probability_sum_conservative = 0.0;
      bool event_reconstructed = false;
      for (int pattern = 0; pattern < pattern_count; ++pattern) {
        const double probability_nominal = tagPatternProbability(
            eligible_fatjets, pattern, kEpsBbNominal, kFakeBbNominal);
        const double probability_conservative = tagPatternProbability(
            eligible_fatjets, pattern, kEpsBbConservative, kFakeBbConservative);
        probability_sum_nominal += probability_nominal;
        probability_sum_conservative += probability_conservative;

        std::vector<int> passing_fat_indices;
        n_true_fat_pass = n_true_fat_fail = 0;
        n_fake_fat_pass = n_fake_fat_fail = 0;
        for (int index = 0; index < n_ak8_retained; ++index) {
          const bool pass = (pattern & (1 << index)) != 0;
          if (pass) passing_fat_indices.push_back(index);
          if (eligible_fatjets[static_cast<std::size_t>(index)].trueDoubleB()) {
            pass ? ++n_true_fat_pass : ++n_true_fat_fail;
          } else {
            pass ? ++n_fake_fat_pass : ++n_fake_fat_fail;
          }
        }

        std::vector<Jet> available_true;
        std::vector<Jet> available_c;
        std::vector<Jet> available_light;
        for (const Jet& jet : true_bjets) {
          if (!overlapsFat(jet.p4, passing_fat_indices, eligible_fatjets)) {
            available_true.push_back(jet);
          }
        }
        for (const Jet& jet : c_candidates) {
          if (!overlapsFat(jet.p4, passing_fat_indices, eligible_fatjets)) {
            available_c.push_back(jet);
          }
        }
        for (const Jet& jet : light_candidates) {
          if (!overlapsFat(jet.p4, passing_fat_indices, eligible_fatjets)) {
            available_light.push_back(jet);
          }
        }

        const int required_resolved_jets =
            2 * (kHiggsCount - static_cast<int>(passing_fat_indices.size()));
        const int true_population_cap =
            std::max(0, 8 - options.c_mistags - options.light_mistags);
        const int true_to_use = std::min(
            required_resolved_jets,
            std::min(options.max_reco_jets,
                     std::min(true_population_cap,
                              static_cast<int>(available_true.size()))));
        int remaining = required_resolved_jets - true_to_use;
        const int c_to_use = std::min(
            remaining,
            std::min(options.c_mistags, static_cast<int>(available_c.size())));
        remaining -= c_to_use;
        const int light_to_use = std::min(
            remaining,
            std::min(options.light_mistags, static_cast<int>(available_light.size())));
        remaining -= light_to_use;
        if (remaining != 0) {
          ++failed_patterns;
          continue;
        }

        std::vector<Jet> resolved_jets;
        resolved_jets.insert(resolved_jets.end(), available_true.begin(),
                             available_true.begin() + true_to_use);
        resolved_jets.insert(resolved_jets.end(), available_c.begin(),
                             available_c.begin() + c_to_use);
        resolved_jets.insert(resolved_jets.end(), available_light.begin(),
                             available_light.begin() + light_to_use);
        // Fix the resolved-jet ordering before pairing so reconstructed
        // candidate indices stay aligned with their stored source jets.
        std::sort(resolved_jets.begin(), resolved_jets.end(), by_pt);
        BestReconstruction reconstruction = reconstruct(
            resolved_jets, passing_fat_indices, eligible_fatjets);
        if (!reconstruction.valid) {
          ++failed_patterns;
          continue;
        }
        std::sort(reconstruction.candidates.begin(), reconstruction.candidates.end(),
                  [](const Candidate& first, const Candidate& second) {
                    return first.p4.Pt() > second.p4.Pt();
                  });

        hypothesis_index = pattern;
        weight = event_weight;
        n_true_single = true_to_use;
        n_c_mistag = c_to_use;
        n_light_mistag = light_to_use;
        n_merged = static_cast<int>(passing_fat_indices.size());
        category = n_merged == 0 ? 0 : (n_merged <= 2 ? 1 : 2);
        reco_jets_used = static_cast<int>(resolved_jets.size());
        n_configurations = reconstruction.configurations;
        best_score = reconstruction.best_score;
        second_score = std::isfinite(reconstruction.second_score)
            ? reconstruction.second_score : -1.0;
        score_gap = second_score >= 0.0 ? second_score - best_score : -1.0;
        if (n_true_single + n_c_mistag + n_light_mistag !=
            2 * (kHiggsCount - n_merged)) {
          throw std::runtime_error("internal AK4 tag-composition closure failure");
        }

        fillArray(jet_pt, 0.0);
        fillArray(higgs_e, 0.0);
        fillArray(higgs_px, 0.0);
        fillArray(higgs_py, 0.0);
        fillArray(higgs_pz, 0.0);
        fillArray(higgs_mass, 0.0);
        fillArray(higgs_tag_mass, 0.0);
        fillArray(higgs_pt, 0.0);
        fillArray(higgs_y, 0.0);
        fillArray(higgs_type, -1);
        fillArray(higgs_constituent1, -1);
        fillArray(higgs_constituent2, -1);
        fillArray(higgs_constituent1_source, -1);
        fillArray(higgs_constituent2_source, -1);
        fillArray(fat_pt, 0.0);
        fillArray(fat_eta, 0.0);
        fillArray(fat_mass, 0.0);
        fillArray(fat_softdrop_mass, 0.0);
        fillArray(fat_tau21_selected, 0.0);
        fillArray(fat_b_hadron_multiplicity, -1);
        fillArray(fat_c_hadron_multiplicity, -1);
        fillArray(fat_tag_kind, -1);
        fillArray(pair_mass, 0.0);
        fillArray(pair_dr, 0.0);
        fillArray(pair_dy, 0.0);
        fillArray(pair_dphi, 0.0);

        for (std::size_t index = 0;
             index < resolved_jets.size() && index < kPaddedJetCount; ++index) {
          jet_pt[index] = resolved_jets[index].p4.Pt();
        }
        for (std::size_t slot = 0; slot < passing_fat_indices.size(); ++slot) {
          const FatJet& fat = eligible_fatjets[
              static_cast<std::size_t>(passing_fat_indices[slot])];
          fat_pt[slot] = fat.p4.Pt();
          fat_eta[slot] = fat.p4.Eta();
          fat_mass[slot] = fat.p4.M();
          fat_softdrop_mass[slot] = fat.softdrop_p4.M();
          fat_tau21_selected[slot] = fat.tau21;
          fat_b_hadron_multiplicity[slot] = fat.b_hadrons;
          fat_c_hadron_multiplicity[slot] = fat.c_hadrons;
          fat_tag_kind[slot] = fat.trueDoubleB() ? 1 : 0;
        }

        TLorentzVector four_higgs;
        std::vector<Jet> selected_objects = resolved_jets;
        for (int fat_index : passing_fat_indices) {
          Jet object;
          object.p4 = eligible_fatjets[static_cast<std::size_t>(fat_index)].p4;
          object.source = 3;
          selected_objects.push_back(object);
        }
        for (int index = 0; index < kHiggsCount; ++index) {
          const Candidate& candidate = reconstruction.candidates[index];
          higgs_e[index] = candidate.p4.E();
          higgs_px[index] = candidate.p4.Px();
          higgs_py[index] = candidate.p4.Py();
          higgs_pz[index] = candidate.p4.Pz();
          higgs_mass[index] = candidate.p4.M();
          higgs_tag_mass[index] = candidate.tag_mass;
          higgs_pt[index] = candidate.p4.Pt();
          higgs_y[index] = candidate.p4.Rapidity();
          higgs_type[index] = candidate.type;
          higgs_constituent1[index] = candidate.first_jet;
          higgs_constituent2[index] = candidate.second_jet;
          if (candidate.type == 1) {
            higgs_constituent1_source[index] = 3;
          } else {
            higgs_constituent1_source[index] =
                resolved_jets[static_cast<std::size_t>(candidate.first_jet)].source;
            higgs_constituent2_source[index] =
                resolved_jets[static_cast<std::size_t>(candidate.second_jet)].source;
          }
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
            pair_dy[pair_index] = std::fabs(
                reconstruction.candidates[first].p4.Rapidity() -
                reconstruction.candidates[second].p4.Rapidity());
            pair_dphi[pair_index] = absoluteDeltaPhi(
                reconstruction.candidates[first].p4.Phi(),
                reconstruction.candidates[second].p4.Phi());
            ++pair_index;
          }
        }
        m4h = four_higgs.M();
        pt4h = four_higgs.Pt();
        y4h = four_higgs.Rapidity();
        ht = 0.0;
        for (const Jet& object : selected_objects) ht += object.p4.Pt();
        event_centrality = centrality(selected_objects);
        event_sphericity = transverseSphericity(selected_objects);

        tree.Fill();
        event_reconstructed = true;
        hypothesis_row_counter.fill(event_weight);
        category_row_counters[static_cast<std::size_t>(category)].fill(event_weight);
        selected_probability_sum_nominal += event_weight * probability_nominal;
        selected_probability_sum_conservative +=
            event_weight * probability_conservative;
      }
      if (event_reconstructed) events_with_reconstruction.fill(event_weight);
      max_pattern_probability_residual_nominal = std::max(
          max_pattern_probability_residual_nominal,
          std::fabs(probability_sum_nominal - 1.0));
      max_pattern_probability_residual_conservative = std::max(
          max_pattern_probability_residual_conservative,
          std::fabs(probability_sum_conservative - 1.0));
    }

    output.cd();
    tree.Write();
    TNamed observable_schema("observable_schema", "fatjet-ak8-softdrop-v1");
    observable_schema.Write();
    TNamed method_version("method_version", kMethodVersion);
    method_version.Write();
    TNamed preprocessing_version("preprocessing_version", kPreprocessingVersion);
    preprocessing_version.Write();
    TNamed branch_schema("branch_schema_json", branchSchemaJson());
    branch_schema.Write();
    const std::vector<std::string> feature_names = featureNames();
    TNamed feature_names_metadata(
        "feature_names_json", jsonStringArray(feature_names).c_str());
    feature_names_metadata.Write();
    TNamed audit_branch_names(
        "audit_branch_names_json",
        "[\"hypothesis_index\",\"raw_bjets\",\"accepted_bjets\","
        "\"accepted_cjet_candidates\",\"accepted_lightjet_candidates\","
        "\"n_ak8_eligible\",\"n_ak8_hh_diagnostic\","
        "\"n_true_single\",\"n_c_mistag\",\"n_light_mistag\","
        "\"n_true_fat_pass\",\"n_true_fat_fail\","
        "\"n_fake_fat_pass\",\"n_fake_fat_fail\","
        "\"fat_b_hadron_multiplicity\",\"fat_c_hadron_multiplicity\","
        "\"fat_tag_kind\",\"higgs_constituent1_source\","
        "\"higgs_constituent2_source\"]");
    audit_branch_names.Write();
    TNamed category_definition(
        "category_definition_json",
        "{\"0\":\"resolved (zero passing AK8 candidates)\","
        "\"1\":\"mixed (one or two passing AK8 candidates)\","
        "\"2\":\"boosted (three or four passing AK8 candidates)\"}");
    category_definition.Write();
    TNamed pair_order("pair_order_json", "[\"12\",\"13\",\"14\",\"23\",\"24\",\"34\"]");
    pair_order.Write();
    TNamed tagging_definition(
        "tagging_definition_json",
        "{\"efficiencies_applied\":false,\"weight\":\"raw evweight\","
        "\"fat_hypotheses\":\"all pass/fail bitmasks over at most four leading eligible AK8 jets\","
        "\"ak4_closure\":\"n_true_single+n_c_mistag+n_light_mistag=2*(4-n_merged)\"}");
    tagging_definition.Write();
    TParameter<int>("feature_count", static_cast<int>(feature_names.size())).Write();
    TParameter<int>("max_reco_true_bjets", options.max_reco_jets).Write();
    TParameter<int>("c_mistags", options.c_mistags).Write();
    TParameter<int>("light_mistags", options.light_mistags).Write();
    TParameter<int>("smearing_enabled", options.smear ? 1 : 0).Write();
    TParameter<Long64_t>("events_processed", input_counter.events).Write();
    TParameter<Long64_t>("events_reconstructable", events_with_reconstruction.events).Write();
    TParameter<Long64_t>("hypothesis_rows", hypothesis_row_counter.events).Write();
    TParameter<double>("total_weight_in", input_counter.sumw).Write();
    TParameter<double>("total_weight_out", events_with_reconstruction.sumw).Write();
    output.Close();

    std::ofstream summary(options.output_json.c_str());
    if (!summary) {
      throw std::runtime_error("cannot create summary JSON " + options.output_json);
    }
    summary << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"fatjet-ak8-softdrop-v1\",\n"
            << "  \"method_version\": \"" << kMethodVersion << "\",\n"
            << "  \"preprocessing_version\": \"" << kPreprocessingVersion << "\",\n"
            << "  \"input\": \"" << jsonEscape(options.input) << "\",\n"
            << "  \"output_root\": \"" << jsonEscape(options.output_root) << "\",\n"
            << "  \"events_available\": " << available_events << ",\n"
            << "  \"events_requested\": " << events_to_run << ",\n"
            << "  \"max_reco_true_bjets\": " << options.max_reco_jets << ",\n"
            << "  \"c_mistags\": " << options.c_mistags << ",\n"
            << "  \"light_mistags\": " << options.light_mistags << ",\n"
            << "  \"tag_efficiencies_applied\": false,\n"
            << "  \"input_counter\": ";
    writeCounter(summary, input_counter, 0);
    summary << ",\n  \"reconstructable_counter\": ";
    writeCounter(summary, events_with_reconstruction, 0);
    summary << ",\n  \"hypothesis_row_counter\": ";
    writeCounter(summary, hypothesis_row_counter, 0);
    summary << ",\n  \"smearing\": {"
            << "\"enabled\":" << (options.smear ? "true" : "false")
            << ",\"seed\":" << options.seed
            << ",\"preprocessing_version\":\"" << kPreprocessingVersion << "\""
            << ",\"model_id\":\"" << kSmearingModelId << "\""
            << ",\"fourvector_scaling\":\"uniform_correlated\""
            << ",\"correlated_groomed_ungroomed_scaling\":true"
            << ",\"gaussian_draws_per_physical_jet\":1"
            << ",\"ak4_pt_threshold_gev\":20.0"
            << ",\"ak8_pt_threshold_gev\":300.0"
            << ",\"eta_preselection\":\"finite |eta|<2.5 before smearing\"},\n"
            << "  \"fatjet_definition\": {"
            << "\"algorithm\":\"anti-kt\",\"R\":0.8,"
            << "\"softdrop_beta\":0.0,\"softdrop_zcut\":0.1,"
            << "\"max_retained_candidates\":4,"
            << "\"true_double_b_multiplicities\":[2,3],"
            << "\"four_or_more_b_hadrons\":\"diagnostic_only\"},\n"
            << "  \"tagging_scenarios\": {"
            << "\"nominal\":{\"eps_bb\":0.7225,\"fake_bb\":0.10},"
            << "\"conservative\":{\"eps_bb\":0.30,\"fake_bb\":0.01}},\n"
            << "  \"categories\": {\n";
    const std::array<std::string, 3> category_names = {"resolved", "mixed", "boosted"};
    for (std::size_t index = 0; index < category_names.size(); ++index) {
      summary << "    \"" << category_names[index] << "\": ";
      writeCounter(summary, category_row_counters[index], 0);
      summary << (index + 1 == category_names.size() ? "\n" : ",\n");
    }
    summary << "  },\n"
            << "  \"diagnostics\": {\n"
            << "    \"failed_hypothesis_patterns\": " << failed_patterns << ",\n"
            << "    \"fat_candidate_overflow_events\": "
            << fat_candidate_overflow_events << ",\n"
            << "    \"hh_diagnostic_events\": " << hh_diagnostic_events << ",\n"
            << "    \"max_smearing_mass_scaling_residual_gev\": "
            << max_smearing_mass_scaling_residual << ",\n"
            << "    \"max_pattern_probability_residual_nominal\": "
            << max_pattern_probability_residual_nominal << ",\n"
            << "    \"max_pattern_probability_residual_conservative\": "
            << max_pattern_probability_residual_conservative << ",\n"
            << "    \"selected_raw_weight_times_pattern_probability_nominal\": "
            << selected_probability_sum_nominal << ",\n"
            << "    \"selected_raw_weight_times_pattern_probability_conservative\": "
            << selected_probability_sum_conservative << "\n"
            << "  }\n"
            << "}\n";
    summary.close();

    std::cout << "Processed " << input_counter.events << " generator events; wrote "
              << hypothesis_row_counter.events << " reconstructable hypotheses from "
              << events_with_reconstruction.events << " events\n"
              << "ROOT output: " << options.output_root << "\n"
              << "JSON summary: " << options.output_json << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
