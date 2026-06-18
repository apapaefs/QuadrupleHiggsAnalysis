#include "SHERPA/Single_Events/Signal_Processes.H"

#include "PHASIC++/Process/Process_Base.H"
#include "PHASIC++/Process/MCatNLO_Process.H"
#include "PHASIC++/Process/Single_Process.H"
#include "PHASIC++/Scales/Scale_Setter_Base.H"
#include "PHASIC++/Main/Process_Integrator.H"
#include "METOOLS/SpinCorrelations/Amplitude2_Tensor.H"
#include "METOOLS/SpinCorrelations/Decay_Matrix.H"
#include "METOOLS/SpinCorrelations/Spin_Density.H"
#include "ATOOLS/Org/Run_Parameter.H"
#include "ATOOLS/Org/Scoped_Settings.H"
#include "ATOOLS/Org/MyStrStream.H"
#include "ATOOLS/Math/Random.H"
#include "ATOOLS/Phys/NLO_Types.H"
#include "ATOOLS/Phys/Weight_Info.H"
#include "ATOOLS/Phys/Weights.H"
#include "MODEL/Main/Running_AlphaS.H"
#include <algorithm>
#include <map>
#include <sstream>
#include <vector>

using namespace SHERPA;
using namespace METOOLS;
using namespace ATOOLS;
using namespace PHASIC;
using namespace std;

namespace {

struct LHEColourFlowPlan {
  std::vector<int> m_flow1, m_flow2;
};

void SetLHEEndpoint(LHEColourFlowPlan &flows,const size_t leg,
                    const bool colour,const int tag)
{
  if (colour) flows.m_flow1[leg]=tag;
  else flows.m_flow2[leg]=tag;
}

void SetParticleLHEFlow(Particle *const particle,
                        const LHEColourFlowPlan &flows,
                        const size_t leg,
                        const bool incoming)
{
  if (incoming) {
    particle->SetFlow(1,flows.m_flow2[leg]);
    particle->SetFlow(2,flows.m_flow1[leg]);
  }
  else {
    particle->SetFlow(1,flows.m_flow1[leg]);
    particle->SetFlow(2,flows.m_flow2[leg]);
  }
}

void RemoveEndpoint(std::vector<size_t> &ends,const size_t leg)
{
  std::vector<size_t>::iterator eit(std::find(ends.begin(),ends.end(),leg));
  if (eit!=ends.end()) ends.erase(eit);
}

bool MatchEndpoints(const std::vector<size_t> &colours,
                    const std::vector<size_t> &anticolours,
                    std::vector<int> &used,
                    std::vector<size_t> &match,
                    const size_t pos)
{
  if (pos==colours.size()) return true;
  for (size_t i(0);i<anticolours.size();++i) {
    if (used[i] || anticolours[i]==colours[pos]) continue;
    used[i]=1;
    match[pos]=i;
    if (MatchEndpoints(colours,anticolours,used,match,pos+1)) return true;
    used[i]=0;
  }
  return false;
}

std::string LegVectorString(const std::vector<size_t> &legs)
{
  std::ostringstream os;
  os<<"[";
  for (size_t i(0);i<legs.size();++i) {
    if (i) os<<",";
    os<<legs[i];
  }
  os<<"]";
  return os.str();
}

void PreferFirstFinalQQbarSinglet(LHEColourFlowPlan &flows,
                                  std::map<int,std::vector<size_t> > &colours,
                                  std::map<int,std::vector<size_t> > &anticolours,
                                  const Flavour_Vector &flavs,
                                  const size_t nin,
                                  int &nexttag,
                                  const bool debug)
{
  size_t q(flavs.size()), qb(flavs.size());
  long int qkf(0);
  for (size_t i(nin);i<flavs.size();++i) {
    const long int kf((long int)flavs[i]);
    const long int akf(kf<0 ? -kf : kf);
    if (akf<1 || akf>6) continue;
    if (kf>0 && q==flavs.size()) {
      q=i;
      qkf=akf;
    }
    else if (kf<0 && qb==flavs.size() && (qkf==0 || akf==qkf)) {
      qb=i;
      if (qkf==0) qkf=akf;
    }
    if (q!=flavs.size() && qb!=flavs.size()) break;
  }
  if (q==flavs.size() || qb==flavs.size()) return;

  for (std::map<int,std::vector<size_t> >::iterator cit(colours.begin());
       cit!=colours.end();++cit) {
    const int label(cit->first);
    std::vector<size_t> &cs(cit->second);
    std::vector<size_t> &as(anticolours[label]);
    if (std::find(cs.begin(),cs.end(),q)==cs.end()) continue;
    if (std::find(as.begin(),as.end(),qb)==as.end()) continue;
    const int tag(nexttag++);
    SetLHEEndpoint(flows,q,true,tag);
    SetLHEEndpoint(flows,qb,false,tag);
    RemoveEndpoint(cs,q);
    RemoveEndpoint(as,qb);
    if (debug)
      msg_Info()<<"LHE colour-flow hack: forced first final q/qbar singlet "
                <<"legs "<<q<<","<<qb<<" from sampled colour label "
                <<label<<" -> "<<tag<<"\n";
    return;
  }
}

std::string ColourVectorString(const std::vector<int> &cols)
{
  std::ostringstream os;
  os<<"[";
  for (size_t i(0);i<cols.size();++i) {
    if (i) os<<",";
    os<<cols[i];
  }
  os<<"]";
  return os.str();
}

LHEColourFlowPlan BuildLHEColourFlows(const std::vector<int> &ci,
                                      const std::vector<int> &cj,
                                      const Flavour_Vector &flavs,
                                      const size_t nin,
                                      const bool prefer_decay_singlet,
                                      const bool debug)
{
  LHEColourFlowPlan flows;
  flows.m_flow1.assign(flavs.size(),0);
  flows.m_flow2.assign(flavs.size(),0);
  std::map<int,std::vector<size_t> > colours, anticolours;
  std::map<int,int> labels;
  for (size_t i(0);i<flavs.size();++i) {
    if (ci[i]>0) {
      colours[ci[i]].push_back(i);
      labels[ci[i]]=1;
    }
    if (cj[i]>0) {
      anticolours[cj[i]].push_back(i);
      labels[cj[i]]=1;
    }
  }

  int nexttag(501);
  if (prefer_decay_singlet)
    PreferFirstFinalQQbarSinglet(flows,colours,anticolours,flavs,nin,nexttag,
                                 debug);

  for (std::map<int,int>::const_iterator lit(labels.begin());
       lit!=labels.end();++lit) {
    const int label(lit->first);
    std::vector<size_t> &cs(colours[label]);
    std::vector<size_t> &as(anticolours[label]);
    if (cs.size()!=as.size()) {
      const std::string msg(std::string("LHE colour-flow hack found unmatched "
            "sampled colour label ")+ToString(label)+": "
            +ToString(cs.size())+" colour end(s) and "
            +ToString(as.size())+" anticolour end(s).");
      THROW(fatal_error,msg);
    }
    std::vector<int> used(as.size(),0);
    std::vector<size_t> match(cs.size(),0);
    if (!MatchEndpoints(cs,as,used,match,0)) {
      const std::string msg(std::string("LHE colour-flow hack could not pair "
            "sampled colour label ")+ToString(label)+" without self-connecting "
            "a gluon. Colour legs="+LegVectorString(cs)
            +" anticolour legs="+LegVectorString(as)+".");
      THROW(fatal_error,msg);
    }
    for (size_t i(0);i<cs.size();++i) {
      const int tag(nexttag++);
      SetLHEEndpoint(flows,cs[i],true,tag);
      SetLHEEndpoint(flows,as[match[i]],false,tag);
      if (debug)
        msg_Info()<<"LHE colour-flow hack: sampled label "<<label
                  <<" pairs legs "<<cs[i]<<" -> "<<as[match[i]]
                  <<" as LHE tag "<<tag<<"\n";
    }
  }

  for (size_t i(0);i<flavs.size();++i) {
    if (!flavs[i].Strong()) continue;
    if (flavs[i].IsGluon() &&
        (flows.m_flow1[i]==0 || flows.m_flow2[i]==0 ||
         flows.m_flow1[i]==flows.m_flow2[i])) {
      const std::string msg(std::string("LHE colour-flow hack produced an "
            "invalid gluon flow on leg ")+ToString(i)+": "
            +ToString(flows.m_flow1[i])+","+ToString(flows.m_flow2[i])+".");
      THROW(fatal_error,msg);
    }
    if (flavs[i].IsQuark() &&
        ((flows.m_flow1[i]!=0)==(flows.m_flow2[i]!=0))) {
      const std::string msg(std::string("LHE colour-flow hack produced an "
            "invalid quark flow on leg ")+ToString(i)+": "
            +ToString(flows.m_flow1[i])+","+ToString(flows.m_flow2[i])+".");
      THROW(fatal_error,msg);
    }
  }
  return flows;
}

}

Signal_Processes::Signal_Processes(Matrix_Element_Handler* mehandler)
    : p_mehandler(mehandler), p_remnants(mehandler->GetISR()->GetRemnants()),
      m_overweight(0.0)
{
  m_name="Signal_Processes";
  m_type=eph::Perturbative;
  p_yfshandler = mehandler->GetYFS();
  if (p_remnants[0]==NULL || p_remnants[1]==NULL)
    THROW(critical_error,"No beam remnant handler found.");
  Scoped_Settings mepssettings{
    Settings::GetMainSettings()["MEPS"] };
  m_cmode=mepssettings["CLUSTER_MODE"].Get<int>();
  Scoped_Settings spsettings{ Settings::GetMainSettings()["SP"] };
  m_setcolors = spsettings["SET_COLORS"].SetDefault(false).Get<bool>();
  m_lhecolorhack =
    spsettings["LHE_COLOR_FLOW_HACK"].SetDefault(false).Get<bool>();
  m_lhecolordebug =
    spsettings["LHE_COLOR_FLOW_DEBUG"].SetDefault(false).Get<bool>();
  m_adddocumentation = spsettings["ADD_DOC"].SetDefault(false).Get<bool>();
}

Return_Value::code Signal_Processes::Treat(Blob_List * bloblist)
{
  Blob *blob(bloblist->FindFirst(btp::Signal_Process));
  if (blob && blob->Has(blob_status::needs_signal)) {
    MODEL::as->SetActiveAs(PDF::isr::hard_process);
    while (true) {
      if (m_overweight>0.0) {
        if (m_overweight<ran->Get()) {
          m_overweight=0.0;
          CleanUp();
          continue;
        }
        double overweight(m_overweight-1.0);
        if (!FillBlob(bloblist,blob))
          THROW(fatal_error,"Internal error");
        (*blob)["Trials"]->Set(0.0);
        m_overweight=Max(overweight,0.0);
        return Return_Value::Success;
      }
      if (p_mehandler->GenerateOneEvent() &&
          FillBlob(bloblist,blob)) {
        return Return_Value::Success;
      }
      else return Return_Value::New_Event;
    }
  }
  return Return_Value::Nothing;
}

bool Signal_Processes::FillBlob(Blob_List *const bloblist,Blob *const blob)
{
  DEBUG_FUNC(blob->Id());
  PHASIC::Process_Base *proc(p_mehandler->Process());
  LHEColourFlowPlan lheflows;
  const std::vector<int> *stored_ci(NULL), *stored_cj(NULL);
  const size_t expected_color_entries(proc->NIn()+proc->NOut());
  if (m_lhecolorhack) {
    stored_ci=&proc->Integrator()->LastColorI();
    stored_cj=&proc->Integrator()->LastColorJ();
    if (stored_ci->size()<expected_color_entries ||
        stored_cj->size()<expected_color_entries) {
      const std::string msg(std::string("LHE colour-flow hack requested, "
            "but the sampled Comix colour point is missing or too short: I=")
            +ToString(stored_ci->size())+" J="+ToString(stored_cj->size())
            +" expected at least "+ToString(expected_color_entries)+".");
      THROW(fatal_error,msg);
    }
    if (m_lhecolordebug)
      msg_Info()<<"LHE colour-flow hack: I="<<ColourVectorString(*stored_ci)
                <<" J="<<ColourVectorString(*stored_cj)<<"\n";
    const bool prefer_decay_singlet
      (Settings::GetMainSettings()["LHEF_ASSIGN_MISSING_QQBAR_SINGLET"]
       .SetDefault(false).Get<bool>());
    lheflows=BuildLHEColourFlows(*stored_ci,*stored_cj,proc->Flavours(),
                                 proc->NIn(),prefer_decay_singlet,
                                 m_lhecolordebug);
  }
  blob->SetPosition(Vec4D(0.,0.,0.,0.));
  blob->SetTypeSpec(proc->Parent()->Name());
  Cluster_Amplitude *ampl(NULL);
  if (p_mehandler->HasNLO()==3 &&
      proc->Parent()->Info().m_fi.NLOType()!=nlo_type::lo) {
    MCatNLO_Process* mcatnloproc=dynamic_cast<MCatNLO_Process*>(proc->Parent());
    if (mcatnloproc) {
      if (mcatnloproc->WasSEvent()) {
        blob->SetTypeSpec(proc->Parent()->Name()+"+S");

        if (m_adddocumentation) {
          // If documentation mode is enabled, add disconnected blob of original
          // configuration, e.g. for parton-level stitching samples a posteriori
          Process_Base* bproc = mcatnloproc->BVIProc()->Selected();
          Blob* docblob = bloblist->AddBlob(btp::Unspecified);
          for (unsigned int i=0;i<bproc->NIn();i++) {
            Particle* particle = new Particle(0,bproc->Flavours()[i],
                                              bproc->Integrator()->Momenta()[i]);
            particle->SetNumber(0);
            particle->SetStatus(part_status::documentation);
            particle->SetInfo('m');
            docblob->AddToInParticles(particle);
          }
          for (unsigned int i=bproc->NIn(); i<bproc->NIn()+bproc->NOut();i++) {
            Particle* particle = new Particle(0,bproc->Flavours()[i],
                                              bproc->Integrator()->Momenta()[i]);
            particle->SetNumber(0);
            particle->SetStatus(part_status::documentation);
            particle->SetInfo('M');
            docblob->AddToOutParticles(particle);
          }
        }
      }
      else {
        blob->SetTypeSpec(proc->Parent()->Name()+"+H");
      }
      if (m_setcolors) ampl=mcatnloproc->GetAmplitude();
    }
  }
  else {
    if (m_setcolors) ampl=proc->Get<Single_Process>()->
		       Cluster(proc->Integrator()->Momenta(),m_cmode);
  }
  Vec4D cms = Vec4D(0.,0.,0.,0.);
  for (size_t i=0;i<proc->NIn();i++) cms += proc->Integrator()->Momenta()[i];
  blob->SetCMS(cms);
  blob->DeleteOwnedParticles();
  blob->ClearAllData();
  bool success(true);
  Particle *particle(NULL);
  blob->SetStatus(blob_status::needs_harddecays);
  if (proc->Info().m_nlomode!=nlo_mode::fixedorder)
    blob->AddStatus(blob_status::needs_showers);
  const DecayInfo_Vector &decs(proc->DecayInfos());
  blob->AddData("Decay_Info",new Blob_Data<DecayInfo_Vector>(decs));
  for (unsigned int i=0;i<proc->NIn();i++) {
    // Pass born momenta to in if using YFS
    if(p_yfshandler->Mode()!=YFS::yfsmode::off){
      particle = new Particle(0,proc->Flavours()[i],
  			    p_yfshandler->BornMomenta()[i]);
    }
    else{
      particle = new Particle(0,proc->Flavours()[i],
              proc->Integrator()->Momenta()[i]);
    }
    particle->SetNumber(0);
    particle->SetStatus(part_status::decayed);
    particle->SetInfo('G');
    blob->AddToInParticles(particle);
    if (m_lhecolorhack) {
      SetParticleLHEFlow(particle,lheflows,i,true);
    }
    else if (ampl) {
      particle->SetFlow(1,ampl->Leg(i)->Col().m_j);
      particle->SetFlow(2,ampl->Leg(i)->Col().m_i);
    }
  }
  for (unsigned int i=proc->NIn();
       i<proc->NIn()+proc->NOut();i++) {
    particle = new Particle(0,proc->Flavours()[i],
			    proc->Integrator()->Momenta()[i]);
    particle->SetNumber(0);
    for (size_t j(0);j<decs.size();++j)
      if (decs[j]->m_id&(1<<i)) particle->SetMEId(1<<i);
    particle->SetStatus(part_status::active);
    particle->SetInfo('H');
    blob->AddToOutParticles(particle);
    if (m_lhecolorhack) {
      SetParticleLHEFlow(particle,lheflows,i,false);
    }
    else if (ampl) {
      particle->SetFlow(1,ampl->Leg(i)->Col().m_i);
      particle->SetFlow(2,ampl->Leg(i)->Col().m_j);
    }
  }
  if (ampl && p_mehandler->HasNLO()==3 &&
      proc->Parent()->Info().m_fi.NLOType()!=nlo_type::lo) {
    if (ampl->NLO()&4) {
      blob->AddData("MC@NLO_KT2_Stop",new Blob_Data<double>(0.0));
      blob->AddData("MC@NLO_KT2_Start",new Blob_Data<double>(ampl->MuQ2()));
    }
    else if (ampl->Next()) {
      DEBUG_VAR(*ampl->Next());
      blob->AddData("MC@NLO_KT2_Stop",new Blob_Data<double>(ampl->KT2()));
      blob->AddData("MC@NLO_KT2_Start",new Blob_Data<double>
                    (ampl->Next()->Next()?ampl->Next()->KT2():ampl->MuQ2()));
    }
    blob->AddData("Resummation_Scale",new Blob_Data<double>(ampl->MuQ2()));
  }
  if (ampl) ampl->Delete();
  ATOOLS::Weight_Info winfo(p_mehandler->WeightInfo());
  double weightfactor(1.0);
  if (p_mehandler->EventGenerationMode() == 1) {
    m_overweight = p_mehandler->WeightFactor() - 1.0;
    if (m_overweight < 0.0) {
      m_overweight = 0.0;
    } else {
      weightfactor = 1.0 / (m_overweight + 1.0);
      winfo.m_weightsmap *= weightfactor;
      NLO_subevtlist* nlos=proc->GetSubevtList();
      if (nlos) (*nlos) *= weightfactor;
    }
  }
  if(p_yfshandler->HasFSR()!=0){
    // Add the fsr corrected final states
      Particle_Vector out = blob->GetOutParticles();
      Particle_Vector yfsout = p_yfshandler->m_particles;
      ATOOLS::ParticleMomMap yfsoutMap = p_yfshandler->m_outparticles;
      if(out.size()!=(yfsout.size()-2)){
        msg_Error()<<METHOD<<" Missmatch in outparitcles for YFS"<<std::endl
                            <<"Born Out size = "<< out.size()<<std::endl
                            <<"YFS Out size = "<< yfsout.size()<<std::endl;
      }
      for(int i=0; i<out.size(); i++){
        blob->OutParticle(i)->SetMomentum(yfsoutMap[yfsout[i+2]]); // remove born momenta
      }
    }
  if (p_yfshandler->Mode()!=YFS::yfsmode::off) {
    // blob->SetStatus(blob_status::needs_yfs);
    ATOOLS::Vec4D_Vector isrphotons = p_yfshandler->GetISRPhotons();
    ATOOLS::Vec4D_Vector fsrphotons;
    Particle *particle;
    if (p_yfshandler->HasFSR()) {
      fsrphotons = p_yfshandler->GetFSRPhotons();
    }
    if (p_yfshandler->FillBlob()) {
      for (int i = 0; i < isrphotons.size(); ++i)
      {
        particle = new Particle(-1, Flavour(22),
                                isrphotons[i]);
        particle->SetNumber(0);
        particle->SetInfo('S');
        blob->AddToOutParticles(particle);
      }
      for (int i = 0; i < fsrphotons.size(); ++i)
      {
        particle = new Particle(-1, Flavour(22),
                                fsrphotons[i]);
        particle->SetNumber(0);
        particle->SetInfo('S');
        blob->AddToOutParticles(particle);
      }
    }
    p_yfshandler->SplitPhotons(blob);
  }

  blob->AddData("WeightsMap",new Blob_Data<Weights_Map>(winfo.m_weightsmap));
  blob->AddData("MEWeight",new Blob_Data<double>(winfo.m_dxs));
  blob->AddData("Weight_Norm",new Blob_Data<double>
		(p_mehandler->Sum()*rpa->Picobarn()));
  blob->AddData("Trials",new Blob_Data<double>(winfo.m_ntrial));
  blob->AddData("Enhance",new Blob_Data<double>
                (proc->Integrator()->EnhanceFactor()));
  blob->AddData("Factorisation_Scale",new Blob_Data<double>
                (sqrt(winfo.m_pdf.m_muf12*winfo.m_pdf.m_muf22)));
  blob->AddData("PDFInfo",new Blob_Data<ATOOLS::PDF_Info>(winfo.m_pdf));
  blob->AddData("Orders",new Blob_Data<std::vector<double> >
		(p_mehandler->Process()->MaxOrders()));
  blob->AddData("NLOType",new Blob_Data<std::string>
                (ToString(proc->Info().m_fi.m_nlotype)));
  blob->AddData("NLOOrder",new Blob_Data<std::vector<double> >
                (proc->Info().m_fi.m_nlocpl));

  ME_Weight_Info* wgtinfo=proc->GetMEwgtinfo();
  if (wgtinfo) {
    blob->AddData("MEWeightInfo",new Blob_Data<ME_Weight_Info*>(wgtinfo));
    blob->AddData("Renormalization_Scale",new Blob_Data<double>(wgtinfo->m_mur2));
    blob->AddData("Factorization_Scale",new Blob_Data<double>(wgtinfo->m_muf2));
  }
  NLO_subevtlist* nlos=proc->GetSubevtList();
  if (nlos) blob->AddData("NLO_subeventlist",new Blob_Data<NLO_subevtlist*>(nlos));

  if (rpa->gen.HardSC() || (rpa->gen.SoftSC() && !Flavour(kf_tau).IsStable())) {
    DEBUG_INFO("Filling amplitude tensor for spin correlations.");
    std::vector<Spin_Amplitudes> amps;
    std::vector<std::vector<Complex> > cols;
    proc->FillAmplitudes(amps, cols);
    DEBUG_VAR(amps[0]);
    Particle_Vector inparts=blob->GetInParticles();
    Particle_Vector outparts=blob->GetOutParticles();
    vector<pair<Particle*, size_t> > parts(inparts.size()+outparts.size());
    for (size_t i=0; i<inparts.size(); ++i)
      parts[i]=make_pair(inparts[i], i);
    for (size_t i=inparts.size(); i<inparts.size()+outparts.size(); ++i)
      parts[i]=make_pair(outparts[i-inparts.size()], i);

    DEBUG_INFO("particles before stability sorting:");
    for (size_t i=0; i<parts.size(); ++i) DEBUG_INFO(parts[i].first->RefFlav());
    stable_sort(parts.begin(), parts.end(), Amplitude2_Tensor::SortCrit);
    DEBUG_INFO("particles after stability sorting:");
    for (size_t i=0; i<parts.size(); ++i) DEBUG_INFO(parts[i].first->RefFlav());

    vector<int> permutation(parts.size(), -1);
    for (size_t i=0; i<parts.size(); ++i) permutation[parts[i].second]=i;
    DEBUG_INFO("permutation:");
    for (size_t i=0; i<parts.size(); ++i) DEBUG_INFO(permutation[i]);

    vector<int> spin_i(parts.size(), -1), spin_j(parts.size(), -1);
    vector<Particle*> partsonly(parts.size());
    for (size_t i=0; i<parts.size(); ++i) partsonly[i]=parts[i].first;

    auto atensor = std::make_shared<Amplitude2_Tensor>(partsonly,
                                                       permutation,
                                                       0,
                                                       amps,
                                                       spin_i, spin_j);
    DEBUG_VAR(*atensor);
    blob->AddData("ATensor",
                  new Blob_Data<METOOLS::Amplitude2_Tensor_SP>(atensor));
  }
  if(p_yfshandler->Mode()!=YFS::yfsmode::off){
    p_yfshandler->YFSDebug(p_mehandler->Sum()*rpa->Picobarn());
  }
  return success;
}

void Signal_Processes::CleanUp(const size_t& mode)
{
  if (m_overweight>0.0) return;
}

void Signal_Processes::Finish(const std::string&) {}
