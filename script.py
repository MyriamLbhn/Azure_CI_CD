from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment


# Créez un objet Workspace en utilisant vos informations d'identification Azure
ws = Workspace.get(name="car_price_estimation_2",
                   subscription_id='111aaa69-41b9-4dfd-b6af-2ada039dd1ae',
                   resource_group='RG_LEBIHAN_2')



# Créez un objet ComputeTarget qui représente votre ressource de calcul distante
compute_target = ws.compute_targets['mlebihan2']


# Créez un objet Environment pour spécifier les dépendances de votre script
myenv = Environment.from_pip_requirements(name="myenv", file_path="requirements.txt")



# Créez un objet ScriptRunConfig pour spécifier comment votre script doit être exécuté
src = ScriptRunConfig(source_directory=".",
                      script="train.py",
                      compute_target=compute_target,
                      environment=myenv)


# Créez un objet Experiment pour exécuter votre script d'apprentissage automatique
experiment = Experiment(workspace=ws, name="test_VsCode_Azure")

# Soumettez votre configuration d'exécution à Azure Machine Learning
run = experiment.submit(config=src)

# Attendez que l'exécution soit terminée et affichez la sortie
run.wait_for_completion(show_output=True)