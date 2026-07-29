#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

#include <TFile.h>
#include <TLorentzVector.h>
#include <TRandom3.h>
#include <TTree.h>

namespace {

constexpr int kMaxInputJets = 100;
constexpr double kMinimumSmearedEnergyGeV = 1.0e-6;
constexpr const char* kDefaultAnalysisId = "hhh-hhhh-ge6b-v1";
constexpr const char* kSmearingModelId =
    "cms-energy-uniform-fourvector-v1";
constexpr const char* kSmearingAcceptanceOrder =
    "raw_abs_eta_then_smear_then_smeared_pt";

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
    double total_weight = 0.0;
    double total_weight_squared = 0.0;
    long long invalid_multiplicity_events = 0;
    long long bjet_upward_migrations = 0;
    long long bjet_downward_migrations = 0;
    long long non_bjet_upward_migrations = 0;
    long long non_bjet_downward_migrations = 0;
    double maximum_probability_closure_residual = 0.0;
    std::map<int, long long> truth_multiplicity;
    CategoryAccumulator exact6;
    CategoryAccumulator exact7;
    CategoryAccumulator at_least8;
    CategoryAccumulator at_least6;

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

      int accepted_bjets = 0;
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
        if (smeared_passes) ++accepted_bjets;
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

    std::ofstream output(options.output);
    if (!output) {
      throw std::runtime_error("cannot create JSON output " +
                               options.output);
    }
    output << std::setprecision(17);
    output << "{\n"
           << "  \"format_version\": 1,\n"
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
    writeCategory(output, "ge6", at_least6_result, false);
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
    std::cout << "Wrote " << options.output << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << "\n";
    return 1;
  }
}
