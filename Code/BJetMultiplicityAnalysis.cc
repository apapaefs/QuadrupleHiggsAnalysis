#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <TFile.h>
#include <TLorentzVector.h>
#include <TRandom3.h>
#include <TTree.h>

namespace {

constexpr int kMaxInputJets = 100;
constexpr int kPairingJetCount = 6;
constexpr int kPairingCandidateCount = 3;
constexpr double kMinimumSmearedEnergyGeV = 1.0e-6;
constexpr const char* kDefaultAnalysisId = "hhh-hhhh-ge6b-pairing-v3";
constexpr const char* kSmearingModelId =
    "cms-energy-uniform-fourvector-v1";
constexpr const char* kSmearingAcceptanceOrder =
    "raw_abs_eta_then_smear_then_smeared_pt";
constexpr const char* kPairingScoreId =
    "atlas-3h-l1-mass-residual-v1";
const std::array<double, kPairingCandidateCount> kPairingMassTargetsGeV = {
    {120.0, 115.0, 110.0}};

using Pairing = std::array<int, kPairingJetCount>;
using SixJetIndices = std::array<int, kPairingJetCount>;

struct Options {
  std::string input;
  std::string output;
  std::string process = "unknown";
  std::string analysis_id = kDefaultAnalysisId;
  unsigned int seed = 14101983U;
  Long64_t max_events = 0;
  double pt_cut_gev = 20.0;
  double eta_cut = 2.5;
  double btag_efficiency = 0.85;
  double pairing_cut_gev = std::numeric_limits<double>::quiet_NaN();
  double pairing_target_efficiency =
      std::numeric_limits<double>::quiet_NaN();
  bool smear = true;
};

struct CategoryAccumulator {
  double probability_sum = 0.0;
  double weighted_sum = 0.0;
  double sum_w2_probability = 0.0;
  double sum_w2_probability2 = 0.0;

  void fill(double weight, double probability) {
    probability_sum += probability;
    weighted_sum += weight * probability;
    const double weight2 = weight * weight;
    sum_w2_probability += weight2 * probability;
    sum_w2_probability2 += weight2 * probability * probability;
  }
};

struct CategoryResult {
  double probability_sum = 0.0;
  double weighted_sum = 0.0;
  double acceptance = 0.0;
  double acceptance_stat_error = 0.0;
};

struct PairingCalibrationEntry {
  double score_gev = 0.0;
  double weighted_probability = 0.0;
};

struct PairingCalibrationResult {
  double cut_gev = std::numeric_limits<double>::quiet_NaN();
  double achieved_efficiency = std::numeric_limits<double>::quiet_NaN();
  double entry_weight_sum = 0.0;
  double target_weight = 0.0;
};

struct PairKinematics {
  std::vector<std::vector<double>> mass_gev;
  std::vector<std::vector<double>> pt_gev;
};

struct PairCandidate {
  double mass_gev = 0.0;
  double pt_gev = 0.0;
  int first = -1;
  int second = -1;
};

void usage(std::ostream& stream) {
  stream
      << "Usage: BJetMultiplicityAnalysis INPUT.root --output result.json "
         "[options]\n"
      << "Options:\n"
      << "  --process NAME             Sample label stored in the output\n"
      << "  --analysis-id ID           Configuration identifier\n"
      << "  --seed N                   Smearing seed (default 14101983)\n"
      << "  --max-events N             Process at most N entries (0 means all)\n"
      << "  --pt-cut GEV               Smeared b-jet pT cut (default 20)\n"
      << "  --eta-cut VALUE            Raw absolute-eta cut (default 2.5)\n"
      << "  --btag-efficiency VALUE    Analytic per-b-jet efficiency (default 0.85)\n"
      << "  --pairing-cut GEV          Apply the fixed ATLAS 3H pairing-score cut\n"
      << "  --pairing-target-efficiency VALUE\n"
      << "                             Calibrate the score cut to this >=6-tag "
         "retention\n"
      << "  --no-smear                 Disable detector smearing\n";
}

std::string requireValue(int argc, char** argv, int& index,
                         const std::string& option) {
  if (index + 1 >= argc) {
    throw std::runtime_error("missing value for " + option);
  }
  ++index;
  return argv[index];
}

Options parseOptions(int argc, char** argv) {
  if (argc < 2) {
    usage(std::cerr);
    throw std::runtime_error("missing input ROOT file");
  }
  Options options;
  options.input = argv[1];
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--output") {
      options.output = requireValue(argc, argv, index, argument);
    } else if (argument == "--process") {
      options.process = requireValue(argc, argv, index, argument);
    } else if (argument == "--analysis-id") {
      options.analysis_id = requireValue(argc, argv, index, argument);
    } else if (argument == "--seed") {
      options.seed = static_cast<unsigned int>(
          std::stoul(requireValue(argc, argv, index, argument)));
    } else if (argument == "--max-events") {
      options.max_events =
          std::stoll(requireValue(argc, argv, index, argument));
    } else if (argument == "--pt-cut") {
      options.pt_cut_gev =
          std::stod(requireValue(argc, argv, index, argument));
    } else if (argument == "--eta-cut") {
      options.eta_cut =
          std::stod(requireValue(argc, argv, index, argument));
    } else if (argument == "--btag-efficiency") {
      options.btag_efficiency =
          std::stod(requireValue(argc, argv, index, argument));
    } else if (argument == "--pairing-cut") {
      options.pairing_cut_gev =
          std::stod(requireValue(argc, argv, index, argument));
    } else if (argument == "--pairing-target-efficiency") {
      options.pairing_target_efficiency =
          std::stod(requireValue(argc, argv, index, argument));
    } else if (argument == "--no-smear") {
      options.smear = false;
    } else if (argument == "-h" || argument == "--help") {
      usage(std::cout);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.output.empty()) {
    throw std::runtime_error("--output is required");
  }
  if (options.max_events < 0) {
    throw std::runtime_error("--max-events must be non-negative");
  }
  if (!(options.pt_cut_gev > 0.0) || !(options.eta_cut > 0.0)) {
    throw std::runtime_error("jet cuts must be positive");
  }
  if (options.btag_efficiency < 0.0 ||
      options.btag_efficiency > 1.0) {
    throw std::runtime_error("--btag-efficiency must be in [0,1]");
  }
  if (std::isfinite(options.pairing_cut_gev) &&
      options.pairing_cut_gev < 0.0) {
    throw std::runtime_error("--pairing-cut must be non-negative");
  }
  if (std::isfinite(options.pairing_target_efficiency) &&
      (!(options.pairing_target_efficiency > 0.0) ||
       options.pairing_target_efficiency > 1.0)) {
    throw std::runtime_error(
        "--pairing-target-efficiency must be in (0,1]");
  }
  if (std::isfinite(options.pairing_cut_gev) &&
      std::isfinite(options.pairing_target_efficiency)) {
    throw std::runtime_error(
        "--pairing-cut and --pairing-target-efficiency are mutually "
        "exclusive");
  }
  return options;
}

std::string jsonEscape(const std::string& value) {
  std::ostringstream output;
  for (const char character : value) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        output << character;
    }
  }
  return output.str();
}

double binomialProbability(int trials, int successes, double efficiency) {
  if (successes < 0 || successes > trials) return 0.0;
  if (efficiency == 0.0) return successes == 0 ? 1.0 : 0.0;
  if (efficiency == 1.0) return successes == trials ? 1.0 : 0.0;
  const double logarithm =
      std::lgamma(static_cast<double>(trials) + 1.0) -
      std::lgamma(static_cast<double>(successes) + 1.0) -
      std::lgamma(static_cast<double>(trials - successes) + 1.0) +
      successes * std::log(efficiency) +
      (trials - successes) * std::log1p(-efficiency);
  return std::exp(logarithm);
}

double probabilityAtLeast(int trials, int threshold, double efficiency) {
  if (trials < threshold) return 0.0;
  double below = 0.0;
  for (int successes = 0; successes < threshold; ++successes) {
    below += binomialProbability(trials, successes, efficiency);
  }
  return std::max(0.0, std::min(1.0, 1.0 - below));
}

void generatePairingsRecursive(std::vector<int>& items, int start,
                               std::vector<Pairing>& pairings) {
  if (start == static_cast<int>(items.size())) {
    Pairing pairing = {};
    std::copy(items.begin(), items.end(), pairing.begin());
    pairings.push_back(pairing);
    return;
  }
  for (int partner = start + 1;
       partner < static_cast<int>(items.size()); ++partner) {
    std::swap(items[start + 1], items[partner]);
    generatePairingsRecursive(items, start + 2, pairings);
    std::swap(items[start + 1], items[partner]);
  }
}

std::vector<Pairing> makePairings() {
  std::vector<int> items(kPairingJetCount);
  std::iota(items.begin(), items.end(), 0);
  std::vector<Pairing> pairings;
  generatePairingsRecursive(items, 0, pairings);
  return pairings;
}

template <typename Callback>
void enumerateSixJetCombinationsRecursive(
    int jet_count, int next_index, int depth, SixJetIndices& indices,
    Callback& callback) {
  if (depth == kPairingJetCount) {
    callback(indices);
    return;
  }
  const int remaining = kPairingJetCount - depth;
  for (int index = next_index; index <= jet_count - remaining; ++index) {
    indices[depth] = index;
    enumerateSixJetCombinationsRecursive(
        jet_count, index + 1, depth + 1, indices, callback);
  }
}

template <typename Callback>
void enumerateSixJetCombinations(int jet_count, Callback callback) {
  if (jet_count < kPairingJetCount) return;
  SixJetIndices indices = {};
  enumerateSixJetCombinationsRecursive(
      jet_count, 0, 0, indices, callback);
}

PairKinematics pairKinematics(
    const std::vector<TLorentzVector>& jets) {
  PairKinematics kinematics;
  kinematics.mass_gev.assign(
      jets.size(), std::vector<double>(jets.size(), 0.0));
  kinematics.pt_gev.assign(
      jets.size(), std::vector<double>(jets.size(), 0.0));
  for (std::size_t first = 0; first < jets.size(); ++first) {
    for (std::size_t second = first + 1; second < jets.size(); ++second) {
      const TLorentzVector candidate = jets[first] + jets[second];
      const double mass_gev = candidate.M();
      const double pt_gev = candidate.Pt();
      if (!std::isfinite(mass_gev) || !std::isfinite(pt_gev)) {
        throw std::runtime_error(
            "encountered non-finite dijet kinematics in pairing");
      }
      kinematics.mass_gev[first][second] = mass_gev;
      kinematics.mass_gev[second][first] = mass_gev;
      kinematics.pt_gev[first][second] = pt_gev;
      kinematics.pt_gev[second][first] = pt_gev;
    }
  }
  return kinematics;
}

double pairingScore(
    const SixJetIndices& indices,
    const PairKinematics& pair_kinematics,
    const std::vector<Pairing>& pairings) {
  double best = std::numeric_limits<double>::infinity();
  for (const Pairing& pairing : pairings) {
    std::array<PairCandidate, kPairingCandidateCount> candidates = {};
    for (int candidate_index = 0;
         candidate_index < kPairingCandidateCount; ++candidate_index) {
      const int first =
          indices[pairing[2 * candidate_index]];
      const int second =
          indices[pairing[2 * candidate_index + 1]];
      candidates[candidate_index] = {
          pair_kinematics.mass_gev[first][second],
          pair_kinematics.pt_gev[first][second],
          std::min(first, second),
          std::max(first, second)};
    }
    std::sort(
        candidates.begin(), candidates.end(),
        [](const PairCandidate& left, const PairCandidate& right) {
          if (left.pt_gev != right.pt_gev) {
            return left.pt_gev > right.pt_gev;
          }
          if (left.first != right.first) {
            return left.first < right.first;
          }
          return left.second < right.second;
        });
    double score = 0.0;
    for (int candidate_index = 0;
         candidate_index < kPairingCandidateCount; ++candidate_index) {
      score += std::fabs(
          candidates[candidate_index].mass_gev -
          kPairingMassTargetsGeV[candidate_index]);
    }
    best = std::min(best, score);
  }
  return best;
}

double topSixCombinationProbability(int sixth_jet_rank,
                                    double efficiency) {
  if (sixth_jet_rank < kPairingJetCount - 1) return 0.0;
  const int higher_untagged =
      sixth_jet_rank - (kPairingJetCount - 1);
  return std::pow(efficiency, kPairingJetCount) *
         std::pow(1.0 - efficiency, higher_untagged);
}

PairingCalibrationResult calibratePairingCut(
    std::vector<PairingCalibrationEntry> entries,
    double target_efficiency, double weighted_ge6_sum) {
  if (!(weighted_ge6_sum > 0.0) || !std::isfinite(weighted_ge6_sum)) {
    throw std::runtime_error(
        "cannot calibrate pairing cut with nonpositive >=6-tag weight");
  }
  if (entries.empty()) {
    throw std::runtime_error(
        "cannot calibrate pairing cut without score entries");
  }
  for (const PairingCalibrationEntry& entry : entries) {
    if (!std::isfinite(entry.score_gev) ||
        !std::isfinite(entry.weighted_probability) ||
        entry.weighted_probability < 0.0) {
      throw std::runtime_error(
          "pairing calibration requires finite nonnegative weights");
    }
  }
  std::sort(
      entries.begin(), entries.end(),
      [](const PairingCalibrationEntry& left,
         const PairingCalibrationEntry& right) {
        return left.score_gev < right.score_gev;
      });
  PairingCalibrationResult result;
  for (const PairingCalibrationEntry& entry : entries) {
    result.entry_weight_sum += entry.weighted_probability;
  }
  const double closure = std::fabs(
      result.entry_weight_sum - weighted_ge6_sum);
  if (closure >
      std::max(1.0e-12, 1.0e-10 * std::fabs(weighted_ge6_sum))) {
    throw std::runtime_error(
        "pairing calibration entries do not close to the >=6-tag weight");
  }
  result.target_weight = target_efficiency * weighted_ge6_sum;
  double cumulative = 0.0;
  for (const PairingCalibrationEntry& entry : entries) {
    cumulative += entry.weighted_probability;
    if (cumulative >= result.target_weight) {
      result.cut_gev = entry.score_gev;
      break;
    }
  }
  if (!std::isfinite(result.cut_gev)) {
    throw std::runtime_error("failed to determine pairing calibration cut");
  }
  cumulative = 0.0;
  for (const PairingCalibrationEntry& entry : entries) {
    if (entry.score_gev <= result.cut_gev) {
      cumulative += entry.weighted_probability;
    }
  }
  result.achieved_efficiency = cumulative / weighted_ge6_sum;
  return result;
}

TLorentzVector smearJet(const TLorentzVector& input, TRandom3& random,
                        bool enabled) {
  if (!enabled) return input;
  const double energy = input.E();
  if (!std::isfinite(energy) || energy <= 0.0) {
    throw std::runtime_error(
        "cannot smear a jet with non-finite or non-positive energy");
  }
  const double eta = input.Eta();
  double sigma_energy = 0.0;
  if (std::fabs(eta) <= 3.0) {
    sigma_energy =
        std::sqrt(std::pow(0.05 * energy, 2) +
                  energy * std::pow(1.5, 2));
  } else if (std::fabs(eta) <= 5.0) {
    sigma_energy =
        std::sqrt(std::pow(0.130 * energy, 2) +
                  energy * std::pow(2.7, 2));
  }
  const double smeared_energy =
      std::max(kMinimumSmearedEnergyGeV,
               energy + random.Gaus(0.0, sigma_energy));
  const double scale = smeared_energy / energy;
  TLorentzVector output;
  output.SetPxPyPzE(scale * input.Px(), scale * input.Py(),
                   scale * input.Pz(), smeared_energy);
  return output;
}

CategoryResult resultFor(const CategoryAccumulator& accumulator,
                         double total_weight, double total_weight_squared,
                         Long64_t event_count) {
  CategoryResult result;
  result.probability_sum = accumulator.probability_sum;
  result.weighted_sum = accumulator.weighted_sum;
  if (total_weight == 0.0) return result;
  result.acceptance = accumulator.weighted_sum / total_weight;
  double residual =
      accumulator.sum_w2_probability2 -
      2.0 * result.acceptance * accumulator.sum_w2_probability +
      result.acceptance * result.acceptance * total_weight_squared;
  residual = std::max(0.0, residual);
  if (event_count > 1) {
    residual *=
        static_cast<double>(event_count) /
        static_cast<double>(event_count - 1);
  }
  result.acceptance_stat_error =
      std::sqrt(residual) / std::fabs(total_weight);
  return result;
}

void writeCategory(std::ostream& output, const std::string& name,
                   const CategoryResult& result, bool trailing_comma) {
  output << "    \"" << name << "\": {\n"
         << "      \"probability_sum\": " << result.probability_sum << ",\n"
         << "      \"weighted_sum\": " << result.weighted_sum << ",\n"
         << "      \"acceptance\": " << result.acceptance << ",\n"
         << "      \"acceptance_stat_error\": "
         << result.acceptance_stat_error << "\n"
         << "    }" << (trailing_comma ? "," : "") << "\n";
}

long long fileSize(const std::string& path) {
  struct stat info {};
  if (stat(path.c_str(), &info) != 0) return -1;
  return static_cast<long long>(info.st_size);
}

long long fileMtime(const std::string& path) {
  struct stat info {};
  if (stat(path.c_str(), &info) != 0) return -1;
  return static_cast<long long>(info.st_mtime);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parseOptions(argc, argv);
    TFile input(options.input.c_str(), "READ");
    if (input.IsZombie()) {
      throw std::runtime_error("cannot open ROOT input " + options.input);
    }
    TTree* tree = dynamic_cast<TTree*>(input.Get("Data"));
    if (tree == nullptr) {
      throw std::runtime_error("input does not contain a Data tree");
    }
    for (const char* branch :
         {"evweight", "numbJets", "thebJets", "numJets", "theJets"}) {
      if (tree->GetBranch(branch) == nullptr) {
        throw std::runtime_error(std::string("missing required branch ") +
                                 branch);
      }
    }

    double event_weight = 0.0;
    int number_bjets = 0;
    int number_jets = 0;
    double bjets[5][kMaxInputJets] = {};
    double jets[5][kMaxInputJets] = {};
    tree->SetBranchAddress("evweight", &event_weight);
    tree->SetBranchAddress("numbJets", &number_bjets);
    tree->SetBranchAddress("thebJets", &bjets);
    tree->SetBranchAddress("numJets", &number_jets);
    tree->SetBranchAddress("theJets", &jets);

    const Long64_t available_events = tree->GetEntries();
    const Long64_t events_to_process =
        options.max_events > 0
            ? std::min(options.max_events, available_events)
            : available_events;
    if (events_to_process <= 0) {
      throw std::runtime_error("input Data tree contains no events");
    }

    TRandom3 random(options.seed);
    const std::vector<Pairing> pairings = makePairings();
    if (pairings.size() != 15U) {
      throw std::runtime_error(
          "failed to generate the 15 canonical six-jet pairings");
    }
    const bool apply_pairing_cut =
        std::isfinite(options.pairing_cut_gev);
    const bool calibrate_pairing_cut =
        std::isfinite(options.pairing_target_efficiency);
    double total_weight = 0.0;
    double total_weight_squared = 0.0;
    long long invalid_multiplicity_events = 0;
    long long bjet_upward_migrations = 0;
    long long bjet_downward_migrations = 0;
    long long non_bjet_upward_migrations = 0;
    long long non_bjet_downward_migrations = 0;
    long long pairing_score_evaluations = 0;
    int maximum_accepted_bjets = 0;
    double maximum_probability_closure_residual = 0.0;
    double maximum_top6_ge6_probability_closure_residual = 0.0;
    double maximum_top6_component_probability_closure_residual = 0.0;
    std::map<int, long long> truth_multiplicity;
    std::vector<PairingCalibrationEntry> pairing_calibration_entries;
    CategoryAccumulator exact6;
    CategoryAccumulator exact7;
    CategoryAccumulator at_least8;
    CategoryAccumulator at_least6;
    CategoryAccumulator paired_exact6;
    CategoryAccumulator paired_exact7;
    CategoryAccumulator paired_at_least8;
    CategoryAccumulator paired_at_least6;

    for (Long64_t event = 0; event < events_to_process; ++event) {
      tree->GetEntry(event);
      total_weight += event_weight;
      total_weight_squared += event_weight * event_weight;

      if (number_bjets < 0 || number_bjets > kMaxInputJets ||
          number_jets < 0 || number_jets > kMaxInputJets) {
        ++invalid_multiplicity_events;
      }
      const int safe_number_bjets =
          std::max(0, std::min(number_bjets, kMaxInputJets));
      const int safe_number_jets =
          std::max(0, std::min(number_jets, kMaxInputJets));

      std::vector<TLorentzVector> accepted_bjet_vectors;
      accepted_bjet_vectors.reserve(safe_number_bjets);
      for (int index = 0; index < safe_number_bjets; ++index) {
        TLorentzVector raw;
        raw.SetPxPyPzE(bjets[1][index], bjets[2][index],
                       bjets[3][index], bjets[0][index]);
        const double raw_eta = raw.Eta();
        const double raw_pt = raw.Pt();
        if (!std::isfinite(raw_eta) || !std::isfinite(raw_pt) ||
            std::fabs(raw_eta) >= options.eta_cut) {
          continue;
        }
        const TLorentzVector smeared =
            smearJet(raw, random, options.smear);
        const bool raw_passes = raw_pt > options.pt_cut_gev;
        const bool smeared_passes =
            std::isfinite(smeared.Pt()) &&
            smeared.Pt() > options.pt_cut_gev;
        if (!raw_passes && smeared_passes) ++bjet_upward_migrations;
        if (raw_passes && !smeared_passes) ++bjet_downward_migrations;
        if (smeared_passes) accepted_bjet_vectors.push_back(smeared);
      }

      // Preserve the extended-v2 random-number stream: every eta-accepted
      // stored non-b jet receives one detector-response draw after all b-jets,
      // even though mistags are disabled for this analysis.
      for (int index = 0; index < safe_number_jets; ++index) {
        TLorentzVector raw;
        raw.SetPxPyPzE(jets[1][index], jets[2][index],
                       jets[3][index], jets[0][index]);
        const double raw_eta = raw.Eta();
        const double raw_pt = raw.Pt();
        if (!std::isfinite(raw_eta) || !std::isfinite(raw_pt) ||
            std::fabs(raw_eta) >= options.eta_cut) {
          continue;
        }
        const TLorentzVector smeared =
            smearJet(raw, random, options.smear);
        const bool raw_passes = raw_pt > options.pt_cut_gev;
        const bool smeared_passes =
            std::isfinite(smeared.Pt()) &&
            smeared.Pt() > options.pt_cut_gev;
        if (!raw_passes && smeared_passes) ++non_bjet_upward_migrations;
        if (raw_passes && !smeared_passes) ++non_bjet_downward_migrations;
      }

      std::stable_sort(
          accepted_bjet_vectors.begin(), accepted_bjet_vectors.end(),
          [](const TLorentzVector& left, const TLorentzVector& right) {
            return left.Pt() > right.Pt();
          });
      const int accepted_bjets =
          static_cast<int>(accepted_bjet_vectors.size());
      maximum_accepted_bjets =
          std::max(maximum_accepted_bjets, accepted_bjets);
      ++truth_multiplicity[accepted_bjets];
      const double probability6 =
          binomialProbability(accepted_bjets, 6,
                              options.btag_efficiency);
      const double probability7 =
          binomialProbability(accepted_bjets, 7,
                              options.btag_efficiency);
      const double probability8plus =
          probabilityAtLeast(accepted_bjets, 8,
                             options.btag_efficiency);
      const double probability6plus =
          probabilityAtLeast(accepted_bjets, 6,
                             options.btag_efficiency);
      maximum_probability_closure_residual =
          std::max(maximum_probability_closure_residual,
                   std::fabs(probability6plus -
                             probability6 - probability7 -
                             probability8plus));
      exact6.fill(event_weight, probability6);
      exact7.fill(event_weight, probability7);
      at_least8.fill(event_weight, probability8plus);
      at_least6.fill(event_weight, probability6plus);

      double paired_probability6 = 0.0;
      double paired_probability7 = 0.0;
      double paired_probability8plus = 0.0;
      double paired_probability6plus = 0.0;
      double top6_probability_sum = 0.0;
      double top6_probability6_sum = 0.0;
      double top6_probability7_sum = 0.0;
      double top6_probability8plus_sum = 0.0;
      if (accepted_bjets >= kPairingJetCount) {
        const PairKinematics pair_kinematics =
            pairKinematics(accepted_bjet_vectors);
        enumerateSixJetCombinations(
            accepted_bjets, [&](const SixJetIndices& indices) {
              ++pairing_score_evaluations;
              const int sixth_jet_rank =
                  indices[kPairingJetCount - 1];
              const int lower_jet_count =
                  accepted_bjets - sixth_jet_rank - 1;
              const double top6_probability =
                  topSixCombinationProbability(
                      sixth_jet_rank, options.btag_efficiency);
              const double top6_probability6 =
                  top6_probability *
                  binomialProbability(
                      lower_jet_count, 0,
                      options.btag_efficiency);
              const double top6_probability7 =
                  top6_probability *
                  binomialProbability(
                      lower_jet_count, 1,
                      options.btag_efficiency);
              const double top6_probability8plus = std::max(
                  0.0, top6_probability -
                           top6_probability6 -
                           top6_probability7);
              top6_probability_sum += top6_probability;
              top6_probability6_sum += top6_probability6;
              top6_probability7_sum += top6_probability7;
              top6_probability8plus_sum += top6_probability8plus;

              const double score_gev =
                  pairingScore(indices, pair_kinematics, pairings);
              if (calibrate_pairing_cut &&
                  top6_probability > 0.0) {
                pairing_calibration_entries.push_back(
                    {score_gev, event_weight * top6_probability});
              }
              if (apply_pairing_cut &&
                  score_gev <= options.pairing_cut_gev) {
                paired_probability6 += top6_probability6;
                paired_probability7 += top6_probability7;
                paired_probability8plus +=
                    top6_probability8plus;
                paired_probability6plus += top6_probability;
              }
            });
      }
      maximum_top6_ge6_probability_closure_residual = std::max(
          maximum_top6_ge6_probability_closure_residual,
          std::fabs(top6_probability_sum - probability6plus));
      maximum_top6_component_probability_closure_residual = std::max(
          maximum_top6_component_probability_closure_residual,
          std::fabs(top6_probability6_sum - probability6) +
              std::fabs(top6_probability7_sum - probability7) +
              std::fabs(top6_probability8plus_sum -
                        probability8plus));
      if (apply_pairing_cut) {
        paired_exact6.fill(event_weight, paired_probability6);
        paired_exact7.fill(event_weight, paired_probability7);
        paired_at_least8.fill(
            event_weight, paired_probability8plus);
        paired_at_least6.fill(
            event_weight, paired_probability6plus);
      }
    }

    if (invalid_multiplicity_events != 0) {
      throw std::runtime_error(
          "encountered out-of-range stored jet multiplicities");
    }
    if (!std::isfinite(total_weight) || total_weight == 0.0) {
      throw std::runtime_error(
          "total input event weight is zero or non-finite");
    }

    const CategoryResult exact6_result =
        resultFor(exact6, total_weight, total_weight_squared,
                  events_to_process);
    const CategoryResult exact7_result =
        resultFor(exact7, total_weight, total_weight_squared,
                  events_to_process);
    const CategoryResult at_least8_result =
        resultFor(at_least8, total_weight, total_weight_squared,
                  events_to_process);
    const CategoryResult at_least6_result =
        resultFor(at_least6, total_weight, total_weight_squared,
                  events_to_process);
    const CategoryResult paired_exact6_result =
        resultFor(paired_exact6, total_weight, total_weight_squared,
                  events_to_process);
    const CategoryResult paired_exact7_result =
        resultFor(paired_exact7, total_weight, total_weight_squared,
                  events_to_process);
    const CategoryResult paired_at_least8_result =
        resultFor(paired_at_least8, total_weight, total_weight_squared,
                  events_to_process);
    const CategoryResult paired_at_least6_result =
        resultFor(paired_at_least6, total_weight, total_weight_squared,
                  events_to_process);
    PairingCalibrationResult pairing_calibration;
    if (calibrate_pairing_cut) {
      pairing_calibration = calibratePairingCut(
          pairing_calibration_entries,
          options.pairing_target_efficiency,
          at_least6_result.weighted_sum);
    }

    std::ofstream output(options.output);
    if (!output) {
      throw std::runtime_error("cannot create JSON output " +
                               options.output);
    }
    output << std::setprecision(17);
    output << "{\n"
           << "  \"format_version\": 2,\n"
           << "  \"analysis_id\": \""
           << jsonEscape(options.analysis_id) << "\",\n"
           << "  \"process\": \"" << jsonEscape(options.process)
           << "\",\n"
           << "  \"input_file\": \"" << jsonEscape(options.input)
           << "\",\n"
           << "  \"input_size_bytes\": " << fileSize(options.input)
           << ",\n"
           << "  \"input_mtime_unix\": " << fileMtime(options.input)
           << ",\n"
           << "  \"available_events\": " << available_events << ",\n"
           << "  \"processed_events\": " << events_to_process << ",\n"
           << "  \"sum_weights\": " << total_weight << ",\n"
           << "  \"sum_weights_squared\": " << total_weight_squared
           << ",\n"
           << "  \"effective_events\": "
           << total_weight * total_weight / total_weight_squared
           << ",\n"
           << "  \"jet_pt_cut_gev\": " << options.pt_cut_gev << ",\n"
           << "  \"jet_abs_eta_cut\": " << options.eta_cut << ",\n"
           << "  \"btag_efficiency\": "
           << options.btag_efficiency << ",\n"
           << "  \"c_mistag_rate\": 0,\n"
           << "  \"light_mistag_rate\": 0,\n"
           << "  \"smearing_enabled\": "
           << (options.smear ? "true" : "false") << ",\n"
           << "  \"smearing_model_id\": \"" << kSmearingModelId
           << "\",\n"
           << "  \"smearing_acceptance_order\": \""
           << kSmearingAcceptanceOrder << "\",\n"
           << "  \"smearing_seed\": " << options.seed << ",\n"
           << "  \"bjet_upward_pt_migrations\": "
           << bjet_upward_migrations << ",\n"
           << "  \"bjet_downward_pt_migrations\": "
           << bjet_downward_migrations << ",\n"
           << "  \"non_bjet_upward_pt_migrations\": "
           << non_bjet_upward_migrations << ",\n"
           << "  \"non_bjet_downward_pt_migrations\": "
           << non_bjet_downward_migrations << ",\n"
           << "  \"maximum_probability_closure_residual\": "
           << maximum_probability_closure_residual << ",\n"
           << "  \"maximum_top6_ge6_probability_closure_residual\": "
           << maximum_top6_ge6_probability_closure_residual << ",\n"
           << "  \"maximum_top6_component_probability_closure_residual\": "
           << maximum_top6_component_probability_closure_residual
           << ",\n"
           << "  \"pairing\": {\n"
           << "    \"score_id\": \"" << kPairingScoreId << "\",\n"
           << "    \"definition\": "
              "\"min_pairings(sum_i |m_bb_i-target_i|)\",\n"
           << "    \"mass_targets_gev\": [120, 115, 110],\n"
           << "    \"selected_tagged_jets\": "
           << kPairingJetCount << ",\n"
           << "    \"selected_jet_order\": "
              "\"six highest-smeared-pT tagged b-jets\",\n"
           << "    \"target_assignment\": "
              "\"candidate pairs ordered by descending pair pT; "
              "targets 120, 115, and 110 GeV assigned in that order\",\n"
           << "    \"canonical_pairings\": " << pairings.size()
           << ",\n"
           << "    \"cut_operator\": \"<=\",\n"
           << "    \"cut_enabled\": "
           << (apply_pairing_cut ? "true" : "false") << ",\n"
           << "    \"cut_gev\": ";
    if (apply_pairing_cut) {
      output << options.pairing_cut_gev;
    } else {
      output << "null";
    }
    output << ",\n"
           << "    \"calibration_mode\": "
           << (calibrate_pairing_cut ? "true" : "false") << ",\n"
           << "    \"calibration_target_efficiency\": ";
    if (calibrate_pairing_cut) {
      output << options.pairing_target_efficiency;
    } else {
      output << "null";
    }
    output << ",\n"
           << "    \"calibrated_cut_gev\": ";
    if (calibrate_pairing_cut) {
      output << pairing_calibration.cut_gev;
    } else {
      output << "null";
    }
    output << ",\n"
           << "    \"calibration_achieved_efficiency\": ";
    if (calibrate_pairing_cut) {
      output << pairing_calibration.achieved_efficiency;
    } else {
      output << "null";
    }
    output << ",\n"
           << "    \"calibration_entry_weight_sum\": ";
    if (calibrate_pairing_cut) {
      output << pairing_calibration.entry_weight_sum;
    } else {
      output << "null";
    }
    output << ",\n"
           << "    \"calibration_target_weight\": ";
    if (calibrate_pairing_cut) {
      output << pairing_calibration.target_weight;
    } else {
      output << "null";
    }
    output << ",\n"
           << "    \"calibration_entries\": "
           << pairing_calibration_entries.size() << ",\n"
           << "    \"pairing_score_evaluations\": "
           << pairing_score_evaluations << ",\n"
           << "    \"maximum_accepted_bjets\": "
           << maximum_accepted_bjets << "\n"
           << "  },\n"
           << "  \"truth_accepted_bjet_multiplicity\": {\n";
    std::size_t multiplicity_index = 0;
    for (const auto& item : truth_multiplicity) {
      output << "    \"" << item.first << "\": " << item.second;
      ++multiplicity_index;
      output << (multiplicity_index < truth_multiplicity.size() ? "," : "")
             << "\n";
    }
    output << "  },\n"
           << "  \"tag_categories\": {\n";
    writeCategory(output, "exact6", exact6_result, true);
    writeCategory(output, "exact7", exact7_result, true);
    writeCategory(output, "ge8", at_least8_result, true);
    writeCategory(
        output, "ge6", at_least6_result, apply_pairing_cut);
    if (apply_pairing_cut) {
      writeCategory(
          output, "paired_exact6", paired_exact6_result, true);
      writeCategory(
          output, "paired_exact7", paired_exact7_result, true);
      writeCategory(
          output, "paired_ge8", paired_at_least8_result, true);
      writeCategory(
          output, "paired_ge6", paired_at_least6_result, false);
    }
    output << "  }\n"
           << "}\n";
    output.close();
    if (!output) {
      throw std::runtime_error("failed while writing JSON output " +
                               options.output);
    }

    std::cout << "Processed " << events_to_process << " event(s); "
              << "weighted >=6-tag acceptance = "
              << at_least6_result.acceptance << " +/- "
              << at_least6_result.acceptance_stat_error << "\n";
    if (apply_pairing_cut) {
      std::cout << "Weighted paired >=6-tag acceptance at score <= "
                << options.pairing_cut_gev << " GeV = "
                << paired_at_least6_result.acceptance << " +/- "
                << paired_at_least6_result.acceptance_stat_error
                << "\n";
    }
    if (calibrate_pairing_cut) {
      std::cout << "Calibrated pairing cut = "
                << pairing_calibration.cut_gev
                << " GeV for target retention "
                << options.pairing_target_efficiency
                << "; achieved "
                << pairing_calibration.achieved_efficiency << "\n";
    }
    std::cout << "Wrote " << options.output << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << "\n";
    return 1;
  }
}
