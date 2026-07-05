"""Herwig card templates for forced g -> b bbar split samples."""

import argparse
from collections import namedtuple
from pathlib import Path


ProcessConfig = namedtuple(
    "ProcessConfig",
    [
        "process_id",
        "min_b",
        "min_split_pairs",
        "require_distinct_hard_gluons",
        "limit_emissions",
        "description",
    ],
)


PROCESS_CONFIGS = {
    "gg_hg": ProcessConfig(
        process_id="gg_hg",
        min_b=2,
        min_split_pairs=1,
        require_distinct_hard_gluons=False,
        limit_emissions="OneFinalStateEmission",
        description="gg -> h + g with one forced final-state g -> b bbar split for hbb validation",
    ),
    "gg_hhhg": ProcessConfig(
        process_id="gg_hhhg",
        min_b=2,
        min_split_pairs=1,
        require_distinct_hard_gluons=False,
        limit_emissions="OneFinalStateEmission",
        description="gg -> hhh + g with one forced final-state g -> b bbar split",
    ),
    "gg_hhgg": ProcessConfig(
        process_id="gg_hhgg",
        min_b=4,
        min_split_pairs=2,
        require_distinct_hard_gluons=True,
        limit_emissions="NoLimit",
        description="gg -> hh + gg with two distinct forced final-state g -> b bbar splits",
    ),
    "gg_hggg": ProcessConfig(
        process_id="gg_hggg",
        min_b=6,
        min_split_pairs=3,
        require_distinct_hard_gluons=True,
        limit_emissions="NoLimit",
        description="gg -> h + ggg with three distinct forced final-state g -> b bbar splits",
    ),
}


FINAL_STATE_SPLITTING_DELETIONS = """\
do SplittingGenerator:DeleteFinalSplitting u->u,g; QtoQGSudakovFSR
do SplittingGenerator:DeleteFinalSplitting d->d,g; QtoQGSudakovFSR
do SplittingGenerator:DeleteFinalSplitting s->s,g; QtoQGSudakovFSR
do SplittingGenerator:DeleteFinalSplitting c->c,g; QtoQGSudakovFSR
do SplittingGenerator:DeleteFinalSplitting b->b,g; QtoQGSudakovFSR
do SplittingGenerator:DeleteFinalSplitting t->t,g; QtoQGSudakovFSR
do SplittingGenerator:DeleteFinalSplitting g->g,g; GtoGGSudakovFSR
do SplittingGenerator:DeleteFinalSplitting g->u,ubar; GtoQQbarSudakovFSR
do SplittingGenerator:DeleteFinalSplitting g->d,dbar; GtoQQbarSudakovFSR
do SplittingGenerator:DeleteFinalSplitting g->s,sbar; GtoQQbarSudakovFSR
do SplittingGenerator:DeleteFinalSplitting g->c,cbar; GtoccbarSudakovFSR
do SplittingGenerator:DeleteFinalSplitting g->t,tbar; GtoQQbarSudakovFSR"""


def _yes_no(value):
    return "Yes" if value else "No"


def _default_correction_file(output_prefix):
    return "%s.force_split.weights" % output_prefix


def _output_location_for_hwsim(output_location):
    output_location = str(output_location)
    if output_location.endswith("/"):
        return output_location
    return output_location + "/"


def stage1_lhewriter_card(
    config,
    input_lhe,
    output_prefix,
    events,
    seed=31122002,
    probe_trials=0,
    correction_file=None,
    reset_after_attempts=100000,
    max_try=100000,
):
    """Return a Herwig Stage-1 card that writes split parton-level LHE events."""
    if correction_file is None:
        correction_file = _default_correction_file(output_prefix)

    return """\
##############################################################
# Stage 1 forced g -> b bbar splitting: {description}
# Higgs decays are kept out of this stage; Stage 2 forces h0 -> b,bbar.
##############################################################
cd /Herwig/EventHandlers
create ThePEG::Cuts /Herwig/Cuts/NoCuts

mkdir LesHouches
cd LesHouches
library LesHouches.so
cd /Herwig/EventHandlers
create ThePEG::LesHouchesFileReader theLHReader LesHouches.so

cd /Herwig/Partons
create ThePEG::LHAPDF thePDFset ThePEGLHAPDF.so

cd /Herwig/EventHandlers
create ThePEG::LesHouchesEventHandler LesHouchesHandler
insert LesHouchesHandler:LesHouchesReaders[0] theLHReader
set LesHouchesHandler:PartonExtractor /Herwig/Partons/PPExtractor
set theLHReader:WeightWarnings false
set LesHouchesHandler:WeightOption VarNegWeight
set LesHouchesHandler:CascadeHandler /Herwig/Shower/ShowerHandler
set LesHouchesHandler:HadronizationHandler NULL
set LesHouchesHandler:DecayHandler NULL
set theLHReader:FileName {input_lhe}

cd /Herwig/Partons
set /Herwig/Partons/thePDFset:PDFName NNPDF23_nlo_as_0119
set /Herwig/Partons/RemnantDecayer:AllowTop Yes
set /Herwig/EventHandlers/theLHReader:PDFA /Herwig/Partons/thePDFset
set /Herwig/EventHandlers/theLHReader:PDFB /Herwig/Partons/thePDFset
set /Herwig/Shower/ShowerHandler:PDFA /Herwig/Partons/thePDFset
set /Herwig/Shower/ShowerHandler:PDFB /Herwig/Partons/thePDFset
set /Herwig/Partons/MPIExtractor:FirstPDF /Herwig/Partons/thePDFset
set /Herwig/Partons/MPIExtractor:SecondPDF /Herwig/Partons/thePDFset
set /Herwig/Partons/thePDFset:RemnantHandler /Herwig/Partons/HadronRemnants

cd /Herwig/Generators
create ThePEG::EventGenerator theGenerator
set theGenerator:RandomNumberGenerator /Herwig/Random
set theGenerator:StandardModelParameters /Herwig/Model
set theGenerator:EventHandler /Herwig/EventHandlers/LesHouchesHandler
set theGenerator:EventHandler:Cuts /Herwig/Cuts/NoCuts
set theGenerator:NumberOfEvents {events}
set theGenerator:RandomNumberGenerator:Seed {seed}
set theGenerator:DebugLevel 0
set theGenerator:PrintEvent 100
set theGenerator:MaxErrors 10000

cd /Herwig/Shower
set ShowerHandler:DoISR No
set ShowerHandler:DoFSR Yes
set ShowerHandler:Interactions QCD
set ShowerHandler:HardEmission None
set ShowerHandler:LimitEmissions {limit_emissions}
set ShowerHandler:TruncatedShower No
set ShowerHandler:UseConstituentMasses No
set ShowerHandler:MPIHandler NULL
# Default DecayInShower is 6,23,24,25.  Drop h0 so Stage-1 b quarks can
# only come from the forced final-state gluon splitting.
erase ShowerHandler:DecayInShower 3

library HwMinBShowerVeto.so
create Herwig::MinBShowerVeto ForceSplitVeto HwMinBShowerVeto.so
set ForceSplitVeto:MinB {min_b}
set ForceSplitVeto:MinSplitPairs {min_split_pairs}
set ForceSplitVeto:RequireDistinctHardGluons {require_distinct}
set ForceSplitVeto:SplitMinBPt 15*GeV
set ForceSplitVeto:SplitMaxBEta 3.0
set ForceSplitVeto:SplitMinDeltaR 0.3
set ForceSplitVeto:SplitMinDeltaRToOtherB 0.3
set ForceSplitVeto:ProbeTrials {probe_trials}
set ForceSplitVeto:CorrectionFile {correction_file}
set ForceSplitVeto:ResetAfterAttempts {reset_after_attempts}
set ForceSplitVeto:Type Primary
set ForceSplitVeto:Behaviour Shower
insert ShowerHandler:FullShowerVetoes 0 ForceSplitVeto
set ShowerHandler:MaxTry {max_try}

# Keep only final-state g -> b,bbar.
{final_state_splitting_deletions}

cd /Herwig/Particles
set b:NominalMass 4.7*GeV
set bbar:NominalMass 4.7*GeV

cd /Herwig/EventHandlers
set LesHouchesHandler:HadronizationHandler NULL
set /Herwig/Analysis/Basics:CheckQuark false
set LesHouchesHandler:DecayHandler NULL
set /Herwig/Shower/ShowerHandler:MPIHandler NULL

cd /Herwig/Analysis
library LHEWriter.so
create Herwig::LHEWriter /Herwig/Analysis/LHEWriter
set /Herwig/Analysis/LHEWriter:SkipBeamRemnants Yes
insert /Herwig/Generators/theGenerator:AnalysisHandlers 0 /Herwig/Analysis/LHEWriter

cd /Herwig/Generators
saverun {output_prefix} theGenerator
""".format(
        description=config.description,
        input_lhe=input_lhe,
        events=int(events),
        seed=int(seed),
        limit_emissions=config.limit_emissions,
        min_b=int(config.min_b),
        min_split_pairs=int(config.min_split_pairs),
        require_distinct=_yes_no(config.require_distinct_hard_gluons),
        probe_trials=int(probe_trials),
        correction_file=correction_file,
        reset_after_attempts=int(reset_after_attempts),
        max_try=int(max_try),
        final_state_splitting_deletions=FINAL_STATE_SPLITTING_DELETIONS,
        output_prefix=output_prefix,
    )


def stage2_hwsim_card(input_lhe, output_location, events, run_name, seed=89968250):
    """Return a Herwig Stage-2 card that decays Higgs bosons and runs HwSim."""
    output_location = _output_location_for_hwsim(output_location)
    return """\
##############################################################
# Stage 2 forced h0 -> b,bbar decay and HwSim analysis.
##############################################################
cd /Herwig/EventHandlers
create ThePEG::Cuts /Herwig/Cuts/NoCuts

mkdir LesHouches
cd LesHouches
library LesHouches.so
cd /Herwig/EventHandlers
create ThePEG::LesHouchesFileReader theLHReader LesHouches.so

cd /Herwig/Partons
create ThePEG::LHAPDF thePDFset ThePEGLHAPDF.so

cd /Herwig/EventHandlers
create ThePEG::LesHouchesEventHandler LesHouchesHandler
insert LesHouchesHandler:LesHouchesReaders[0] theLHReader
set LesHouchesHandler:PartonExtractor /Herwig/Partons/PPExtractor
set theLHReader:WeightWarnings false
set LesHouchesHandler:WeightOption VarNegWeight
set LesHouchesHandler:CascadeHandler /Herwig/Shower/ShowerHandler
set LesHouchesHandler:HadronizationHandler /Herwig/Hadronization/ClusterHadHandler
set LesHouchesHandler:DecayHandler /Herwig/Decays/DecayHandler
set theLHReader:FileName {input_lhe}

cd /Herwig/Partons
set /Herwig/Partons/thePDFset:PDFName NNPDF23_nlo_as_0119
set /Herwig/Partons/RemnantDecayer:AllowTop Yes
set /Herwig/EventHandlers/theLHReader:PDFA /Herwig/Partons/thePDFset
set /Herwig/EventHandlers/theLHReader:PDFB /Herwig/Partons/thePDFset
set /Herwig/Shower/ShowerHandler:PDFA /Herwig/Partons/thePDFset
set /Herwig/Shower/ShowerHandler:PDFB /Herwig/Partons/thePDFset
set /Herwig/Partons/MPIExtractor:FirstPDF /Herwig/Partons/thePDFset
set /Herwig/Partons/MPIExtractor:SecondPDF /Herwig/Partons/thePDFset
set /Herwig/Partons/thePDFset:RemnantHandler /Herwig/Partons/HadronRemnants

cd /Herwig/Generators
create ThePEG::EventGenerator theGenerator
set theGenerator:RandomNumberGenerator /Herwig/Random
set theGenerator:StandardModelParameters /Herwig/Model
set theGenerator:EventHandler /Herwig/EventHandlers/LesHouchesHandler
set theGenerator:EventHandler:Cuts /Herwig/Cuts/NoCuts
set theGenerator:NumberOfEvents {events}
set theGenerator:RandomNumberGenerator:Seed {seed}
set theGenerator:DebugLevel 0
set theGenerator:PrintEvent 100
set theGenerator:MaxErrors 10000

library HwSim.so
create Herwig::HwSim /Herwig/Analysis/HwSim
insert /Herwig/Generators/theGenerator:AnalysisHandlers 0 /Herwig/Analysis/HwSim
set /Herwig/Analysis/HwSim:OutputLocation {output_location}
set /Herwig/Analysis/HwSim:BTaggingMethod GhostBHadrons
set /Herwig/Analysis/HwSim:CharmTagging Yes
set /Herwig/Analysis/HwSim:OnTheFlyAnalysis No
set /Herwig/Analysis/HwSim:SaveObjects No
set /Herwig/Analysis/HwSim:SaveReconstructed Yes
set /Herwig/Analysis/HwSim:JetAlgorithm AntiKt
set /Herwig/Analysis/HwSim:RParameter 0.4
set /Herwig/Analysis/HwSim:PTCutParticles 0.4
set /Herwig/Analysis/HwSim:EtaCutParticles 6.0
set /Herwig/Analysis/HwSim:PTCutJets 10.0
set /Herwig/Analysis/HwSim:EtaCutJets 6.0
set /Herwig/Analysis/HwSim:PTCutElectron 10.0
set /Herwig/Analysis/HwSim:EtaCutElectron 6.0
set /Herwig/Analysis/HwSim:PTCutMuon 10.0
set /Herwig/Analysis/HwSim:EtaCutMuon 6.0
set /Herwig/Analysis/HwSim:PTCutPhoton 10.0
set /Herwig/Analysis/HwSim:EtaCutPhoton 6.0

decaymode h0->b,bbar; 1.0 1 /Herwig/Decays/Hff
do /Herwig/Particles/h0:SelectDecayModes h0->b,bbar;
do /Herwig/Particles/h0:PrintDecayModes

cd /Herwig/Generators
saverun {run_name} theGenerator
""".format(
        input_lhe=input_lhe,
        output_location=output_location,
        events=int(events),
        seed=int(seed),
        run_name=run_name,
    )


def higgs_decay_lhewriter_card(input_lhe, output_prefix, events, seed=44071981):
    """Return a validation card that forces h0 -> b,bbar and writes LHE."""
    return """\
##############################################################
# Validation-only forced h0 -> b,bbar decay and LHE writing.
# No hadronization, MPI, or shower emissions are intended here.
##############################################################
cd /Herwig/EventHandlers
create ThePEG::Cuts /Herwig/Cuts/NoCuts

mkdir LesHouches
cd LesHouches
library LesHouches.so
cd /Herwig/EventHandlers
create ThePEG::LesHouchesFileReader theLHReader LesHouches.so

cd /Herwig/Partons
create ThePEG::LHAPDF thePDFset ThePEGLHAPDF.so

cd /Herwig/EventHandlers
create ThePEG::LesHouchesEventHandler LesHouchesHandler
insert LesHouchesHandler:LesHouchesReaders[0] theLHReader
set LesHouchesHandler:PartonExtractor /Herwig/Partons/PPExtractor
set theLHReader:WeightWarnings false
set LesHouchesHandler:WeightOption VarNegWeight
set LesHouchesHandler:CascadeHandler /Herwig/Shower/ShowerHandler
set LesHouchesHandler:HadronizationHandler NULL
set LesHouchesHandler:DecayHandler /Herwig/Decays/DecayHandler
set theLHReader:FileName {input_lhe}

cd /Herwig/Partons
set /Herwig/Partons/thePDFset:PDFName NNPDF23_nlo_as_0119
set /Herwig/Partons/RemnantDecayer:AllowTop Yes
set /Herwig/EventHandlers/theLHReader:PDFA /Herwig/Partons/thePDFset
set /Herwig/EventHandlers/theLHReader:PDFB /Herwig/Partons/thePDFset
set /Herwig/Shower/ShowerHandler:PDFA /Herwig/Partons/thePDFset
set /Herwig/Shower/ShowerHandler:PDFB /Herwig/Partons/thePDFset
set /Herwig/Partons/MPIExtractor:FirstPDF /Herwig/Partons/thePDFset
set /Herwig/Partons/MPIExtractor:SecondPDF /Herwig/Partons/thePDFset
set /Herwig/Partons/thePDFset:RemnantHandler /Herwig/Partons/HadronRemnants

cd /Herwig/Generators
create ThePEG::EventGenerator theGenerator
set theGenerator:RandomNumberGenerator /Herwig/Random
set theGenerator:StandardModelParameters /Herwig/Model
set theGenerator:EventHandler /Herwig/EventHandlers/LesHouchesHandler
set theGenerator:EventHandler:Cuts /Herwig/Cuts/NoCuts
set theGenerator:NumberOfEvents {events}
set theGenerator:RandomNumberGenerator:Seed {seed}
set theGenerator:DebugLevel 0
set theGenerator:PrintEvent 100
set theGenerator:MaxErrors 10000

cd /Herwig/Shower
set ShowerHandler:DoISR No
set ShowerHandler:DoFSR No
set ShowerHandler:MPIHandler NULL

cd /Herwig/EventHandlers
set LesHouchesHandler:HadronizationHandler NULL
set /Herwig/Analysis/Basics:CheckQuark false
set /Herwig/Shower/ShowerHandler:MPIHandler NULL

cd /Herwig/Analysis
library LHEWriter.so
create Herwig::LHEWriter /Herwig/Analysis/LHEWriter
set /Herwig/Analysis/LHEWriter:SkipBeamRemnants Yes
insert /Herwig/Generators/theGenerator:AnalysisHandlers 0 /Herwig/Analysis/LHEWriter

decaymode h0->b,bbar; 1.0 1 /Herwig/Decays/Hff
do /Herwig/Particles/h0:SelectDecayModes h0->b,bbar;
do /Herwig/Particles/h0:PrintDecayModes

cd /Herwig/Generators
saverun {output_prefix} theGenerator
""".format(
        input_lhe=input_lhe,
        output_prefix=output_prefix,
        events=int(events),
        seed=int(seed),
    )


def _write_or_print(card, card_out):
    if card_out:
        Path(card_out).write_text(card)
    else:
        print(card, end="")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    stage1 = subparsers.add_parser("stage1", help="write a Stage-1 LHEWriter card")
    stage1.add_argument("process", choices=sorted(PROCESS_CONFIGS))
    stage1.add_argument("--input-lhe", required=True)
    stage1.add_argument("--output-prefix", required=True)
    stage1.add_argument("--events", type=int, required=True)
    stage1.add_argument("--seed", type=int, default=31122002)
    stage1.add_argument("--probe-trials", type=int, default=0)
    stage1.add_argument("--correction-file")
    stage1.add_argument("--card-out")

    stage2 = subparsers.add_parser("stage2", help="write a Stage-2 HwSim card")
    stage2.add_argument("--input-lhe", required=True)
    stage2.add_argument("--output-location", required=True)
    stage2.add_argument("--events", type=int, required=True)
    stage2.add_argument("--run-name", required=True)
    stage2.add_argument("--seed", type=int, default=89968250)
    stage2.add_argument("--card-out")

    hdecay = subparsers.add_parser("hdecay-lhe", help="write a validation h0 -> b,bbar LHEWriter card")
    hdecay.add_argument("--input-lhe", required=True)
    hdecay.add_argument("--output-prefix", required=True)
    hdecay.add_argument("--events", type=int, required=True)
    hdecay.add_argument("--seed", type=int, default=44071981)
    hdecay.add_argument("--card-out")

    args = parser.parse_args(argv)
    if args.command == "stage1":
        card = stage1_lhewriter_card(
            PROCESS_CONFIGS[args.process],
            input_lhe=args.input_lhe,
            output_prefix=args.output_prefix,
            events=args.events,
            seed=args.seed,
            probe_trials=args.probe_trials,
            correction_file=args.correction_file,
        )
        _write_or_print(card, args.card_out)
    elif args.command == "stage2":
        card = stage2_hwsim_card(
            input_lhe=args.input_lhe,
            output_location=args.output_location,
            events=args.events,
            run_name=args.run_name,
            seed=args.seed,
        )
        _write_or_print(card, args.card_out)
    elif args.command == "hdecay-lhe":
        card = higgs_decay_lhewriter_card(
            input_lhe=args.input_lhe,
            output_prefix=args.output_prefix,
            events=args.events,
            seed=args.seed,
        )
        _write_or_print(card, args.card_out)
    else:
        parser.error("choose a command: stage1, stage2, or hdecay-lhe")


if __name__ == "__main__":
    main()
