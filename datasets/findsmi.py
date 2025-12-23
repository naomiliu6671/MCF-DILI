import pubchempy as pcp
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def get_smiles_by_drug_name(name):
    try:
        compounds = pcp.get_compounds(name, 'name', limit=1)
        if compounds:
            compound = compounds[0]
            if compound.isomeric_smiles:
                return compound.isomeric_smiles
            else:
                inchi = compound.inchi
                smiles = inchi_to_smiles(inchi)
                if smiles:
                    return smiles
                else:
                    return get_smiles_from_iupac(compound.iupac_name)
        else:
            print(f"No compound found for name: {name}")
            return None
    except Exception as e:
        print(f"Error fetching SMILES for {name}: {e}")
        return None


def inchi_to_smiles(inchi):
    mol = Chem.MolFromInchi(inchi)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return None


def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    uncharger = rdMolStandardize.Uncharger()
    mol = uncharger.uncharge(mol)
    clean_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    return clean_smiles


def get_smiles_from_iupac(iupac_name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{iupac_name}/property/CanonicalSMILES/TXT"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text.strip()
    else:
        return None


data = pd.read_csv('DILIconcern.csv')
dataset = pd.DataFrame(columns=['Compound Name', 'SMILES', 'diliseverity', 'vDILIConcern'])

for i in range(len(data)):
    name = data.iloc[i]['CompoundName']
    type = data.iloc[i]['SeverityClass']
    dili = data.iloc[i]['vDILI-Concern']
    smiles = get_smiles_by_drug_name(name)
    smiles = standardize_smiles(smiles)
    if smiles:
        print(f"{name}: {smiles}")
        dataset.loc[i] = [name, smiles, type, dili]
dataset.to_csv('DILIconcern/raw/DILIconcern.csv', index=False)
