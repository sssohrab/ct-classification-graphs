import os, json
import h5py
import sys
sys.path.insert(0, '../../')
from ct.data import ROOT_DIR


def is_content_eligible(content):
    """
    Specifies whether a CT is eligible for classification, i.e., its status is settled (not recruiting), and it is an
    interventional CT (not observational)
    :param content:
    :return:
    """

    content = content["FullStudy"]["Study"]
    eligible_study = ['Interventional']
    eligible_status = ["Completed", "Terminated", "Withdrawn", "Suspended", "Unknown status"]
    if "DesignModule" not in content["ProtocolSection"].keys():
        return False

    study = content["ProtocolSection"]["DesignModule"]["StudyType"]
    if study.lower() not in ' '.join(eligible_study).lower():
        return False

    status = content["ProtocolSection"]["StatusModule"]["OverallStatus"]

    if status.lower() not in ' '.join(eligible_status).lower():
        return False

    if 'PhaseList' not in content["ProtocolSection"]['DesignModule'].keys():
        return False

    return True


def create_summary_ct_v3(content):
    """
    Summarized a json CT into 9 important fields, and removes the label-sensitive content.
    :param content: A dictionry loaded from api-json of CTGov.
    :return: dictionary
    """

    # Eligibility check:
    if not is_content_eligible(content):
        return None

    content = content["FullStudy"]["Study"]

    # Summary template:
    ct_summary = {'Meta': {'NCTId': None, 'OverallStatus': None, 'Split': None, 'PhaseNormalized': None},
                  'Features':
                      {
                          'Basic': {
                              "SponsorCollaboratorsModule": None,
                              "OversightModule": None,
                              "DescriptionModule": None,
                              "ConditionsModule": None,
                              "DesignModule": None,
                              "ArmsInterventionsModule": None,
                              "OutcomesModule": None,
                              "EligibilityModule": None,
                              "ContactsLocationsModule": None
                          },
                          'BasicToText': None
                      }
                  }

    content = content['ProtocolSection']

    ct_summary['Meta']['NCTId'] = content['IdentificationModule']['NCTId']

    for k in ct_summary['Features']['Basic'].keys():
        if k in content.keys():
            ct_summary['Features']['Basic'][k] = content[k]

    ct_summary['Features']['BasicToText'] = str(ct_summary['Features']['Basic'])

    ct_summary['Meta']['OverallStatus'] = content['StatusModule']['OverallStatus']

    return ct_summary


if __name__ == '__main__':  # Run to generate summarized CT's with useful content and label-sensitive stuff removed:

    nctids_all_path = os.path.join(ROOT_DIR, 'raw/nctids.txt')
    raw_dir = os.path.join(ROOT_DIR, 'raw/api-json/')
    preprocessed_dir = os.path.join(ROOT_DIR, 'preproc')  # Where you summarize (pre-process) CT's.

    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)

    with open(nctids_all_path, 'r') as f_raw:
        for line in f_raw.readlines():
            nctid = line.strip()
            print(nctid)
            nct_source_path = os.path.join(raw_dir, nctid + '.json')
            with open(nct_source_path, 'r') as f_source:
                ct = json.load(f_source)

            ct_summary = create_summary_ct_v3(ct)

            if ct_summary is None:
                continue

            cleaned_path = os.path.join(preprocessed_dir, nctid + '.json')
            with open(cleaned_path, 'w') as f_dest:
                json.dump(ct_summary, f_dest, indent=2)
