# TrkOnlyPIDTrain.py
# Make dataset, train and test artificial neural network for TrackPID without calo info
# Original author: Michael MacKenzie, based on TrkPID
# Date: 2026-06-23

import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import pandas as pd
import tensorflow as tf
import json
from pathlib import Path


def import_evtntuple(filename, list_branches):
    # Import EvenNtuple root file and transform it into an awkward array

    print(f'Importing {filename}...')
    file = uproot.open(filename)
    tree = file["EventNtuple/ntuple"]

    array = tree.arrays(list_branches, library='ak')

    return array


def apply_cut(array, array_mc, particle):
    # Apply cuts on the dataset by iterating over events, tracks and track segments ; only keep a selected number of branches useful for training and testing
    # array: awkward array containing the reco branches
    # array_mc: awkward array containing the monte carlo branches
    # particle: 'e' for conversion electron dataset, 'mu' for cosmic muon dataset

    if particle == "e":
        mc_pdg = 11
        label_particle = 1
    elif particle == "mu":
        mc_pdg = 13
        label_particle = 0

    data_array = []
    for i_evt in range(ak.num(array, axis=0)):  # iterate over events
        if i_evt % 10000 == 0: print(f'Processing event {i_evt} for particle {particle}...')
        evt_it = array[i_evt]
        for i_trk in range(ak.num(evt_it['trk','trk.status'], axis=0)):   # iterate over tracks
            if evt_it['trk','trk.pdg'][i_trk] != 11:    # mask pdg hypothesis
                continue
            if array_mc['trkmcsim','pdg'][i_evt,i_trk,0] != mc_pdg:     #mask mc pdg
                continue
            if evt_it['trkqual','trkqual.result'][i_trk] < 0.2:    # mask TrkQual
                continue
            trk_it = evt_it['trksegs'][i_trk]
            for i_trksegs in range(ak.num(trk_it['sid'], axis=0)):  # iterate over track segments
                trksegs_it = trk_it[i_trksegs]
                if trksegs_it['sid'] != 1:  # mask sid
                    continue
                if trksegs_it['mom','fCoordinates','fZ'] < 0:   # mask downstream
                    continue
                if trksegs_it['mom','mag'] < 80 or trksegs_it['mom','mag'] > 130:    # mask momentum
                    continue

                data_array.append({
                    'trkqual': evt_it['trkqual','trkqual.result'][i_trk],
                    'fitcon' : evt_it['trk', 'trk.fitcon'][i_trk],
                    'nnullambig' : evt_it['trk', 'trk.nnullambig'][i_trk],
                    'nmatactive' : evt_it['trk', 'trk.nmatactive'][i_trk],
                    'nactive' : evt_it['trk', 'trk.nactive'][i_trk],
                    'nhits' : evt_it['trk', 'trk.nhits'][i_trk],
                    'dtdz_slope' : evt_it['trkdtdz_slope'][i_trk],
                    'dtdz_chisq' : evt_it['trkdtdz_chisq'][i_trk],
                    'mom': trksegs_it['mom','mag'],
                    'time': trksegs_it['time'],
                    'pt': np.sqrt(trksegs_it['mom','fCoordinates','fX']**2 + trksegs_it['mom','fCoordinates','fY']**2),
                    'pz': trksegs_it['mom','fCoordinates','fZ'],
                    'edep': evt_it['trkcalohit','trkcalohit.edep'][i_trk],
                    'label': label_particle,
                    })

    df_array = pd.DataFrame(data_array)
    print(df_array)
    print(df_array.describe())

    return df_array


def make_dataset(particle, dataset_name, csv_name):
    # Import EventNtuple data and apply a set of cuts ; the cut dataset with only useful branches is saved as a csv file to be used later in the training and test
    # particle: 'e' for conversion electron dataset ; 'mu' for cosmic muon dataset
    # dataset_name: path to the ROOT file containing the EventNtuple tree
    # csv_name: name of the csv file in which the trimmed dataset will be saved in ; this is used to access the data easier later for training

    branches_reco = ['trk','trkqual','trksegs','trksegpars_lh','trkcalohit', 'trkdtdz_slope', 'trkdtdz_chisq']
    branches_mc = ['trkmc','trkmcsim','trksegsmc']
    array = import_evtntuple(dataset_name, branches_reco)
    array_mc = import_evtntuple(dataset_name, branches_mc)

    # make mc time modulo event time
    array_mc['trksegsmc','time_mod'] = np.mod(array_mc['trksegsmc','time'], 1695)

    # make momentum magnitude branches
    array['trksegs','mom','mag'] = np.sqrt((array['trksegs','mom','fCoordinates','fX'])**2 + (array['trksegs','mom','fCoordinates','fY'])**2 + (array['trksegs','mom','fCoordinates','fZ'])**2)
    array_mc['trksegsmc','mom','mag'] = np.sqrt((array_mc['trksegsmc','mom','fCoordinates','fX'])**2 + (array_mc['trksegsmc','mom','fCoordinates','fY'])**2 + (array_mc['trksegsmc','mom','fCoordinates','fZ'])**2)

    df_array = apply_cut(array, array_mc, particle)

    # Save cut array into csv file
    df_array.to_csv(csv_name, index=True)


def train_model(dataframe):
    # Reduce variable precision for training speed
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    PID_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,), batch_size=32),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')
        ])

    # Setup loss, optimizer, metrics, and early stop condition for the model
    model_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model_optimizer = tf.keras.optimizers.Adam()
    model_metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5), tf.keras.metrics.AUC(from_logits=False)]
    PID_model.compile(loss = model_loss, optimizer = model_optimizer, metrics = model_metrics)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", start_from_epoch = 10, patience = 15, restore_best_weights = True, verbose = 1)

    print(PID_model.summary())

    n_epochs = 500
    train_history = PID_model.fit(dataframe[features], dataframe['label'], epochs = n_epochs, validation_split=0.2, callbacks=[early_stop])
    train_history = train_history.history   # extract the training history (loss as function of epochs)
    # Save model and training history
    PID_model.save("TrkOnlyPID_model.keras")
    with open("train_history.json",'w') as history_file:
        json.dump(train_history, history_file)

    return PID_model


def make_results(model, dataset, dataset_name, threshold = 0.5):
    # Print model performances
    results = model.evaluate(dataset[features], dataset['label'])
    print("\n", dataset_name, "loss,", dataset_name, "accuracy,", dataset_name, "AUC:", results, "\n")

    dataset['prediction'] = model.predict(dataset[features])
    dataset['predict_label'] = (dataset['prediction'] > threshold).astype(int)

    # Create confusion matrix
    confusion_matrix = tf.math.confusion_matrix(dataset['label'], dataset['predict_label'], num_classes=2)
    print("\n Confusion matrix: \n [ True negative (correctly labeled cosmic muons) ; False positive (mislabeled cosmic muons) ] \n [ False negative (mislabeled conversion electrons) ; True positive (correctly labeled conversion electron) ]\n", confusion_matrix)

    true_negative , false_positive = confusion_matrix[0].numpy()
    false_negative , true_positive = confusion_matrix[1].numpy()

    TPR = true_positive / (true_positive + false_negative)
    TNR = true_negative / (true_negative + false_positive)

    print("\n", dataset_name, " dataset results:\n")
    print("True Positive Rate (correctly labeled conversion electrons / all conversion electrons): ", 100*TPR, "%")
    print("True Negative Rate (correcly labeled cosmic muons / all cosmic muons): ", 100*TNR, "%")
    print("False Positive Rate (mislabeled conversion electrons / all conversion electrons): ", 100*(1-TPR), "%")
    print("False Negative Rate (mislabeled cosmic muons / all cosmic muons): ", 100*(1-TNR), "%\n")

    return dataset, results, confusion_matrix

def add_variables(df):
    df['cz']             = df['pz'] / df['mom']
    df['velocity']       = 300.*df['mom']/np.sqrt(df['mom']**2 + 0.511**2)
    df['dtdz_exp']       = 1. / (df['velocity']*df['cz'])
    df['dtdz_ratio']     = df['dtdz_slope'] / df['dtdz_exp']
    df['nActiveFrac']    = df['nactive'] / df['nhits']
    df['nMatActiveFrac'] = df['nmatactive'] / df['nhits']
    df['nNullFrac']      = df['nnullambig'] / df['nactive']
    return df

def plot_dataset(csv_name, particle, figdir):
    df = pd.read_csv(csv_name, index_col=0)

    # plot of training features
    fig,ax = plt.subplots(1,1)
    ax.hist(df['mom'], bins=100)
    ax.set_xlabel("reco mom [MeV]")
    ax.set_title("Reconstructed momentum of "+particle)
    fig.savefig(f'{figdir}mom.png')

def plot_model(model, figdir):
    tf.keras.utils.plot_model(model,
                              to_file=f'{figdir}model.png',
                              show_shapes=True,
                              show_dtype=False,
                              show_layer_names=True,
                              rankdir='TB',
                              expand_nested=True,
                              dpi=96
                              )
    # text-based summary
    model.summary()


def plot_feature(dataset_e, dataset_mu, feature, figdir, scale = 'linear', tag = ''):
    # Plot of a branch
    min_x = min(min(dataset_e[feature]), min(dataset_mu[feature]))
    max_x = max(max(dataset_e[feature]), max(dataset_mu[feature]))

    fig,ax = plt.subplots(1,1)
    ax.hist(dataset_e[feature], color='b', alpha=0.5, range=(min_x,max_x), bins=100, density=True)
    ax.hist(dataset_mu[feature], color='r', alpha=0.5, range=(min_x,max_x), bins=100, density=True)

    ax.set_xlabel(feature)
    ax.set_xlim(min_x-0.05*(max_x-min_x), max_x+0.05*(max_x-min_x))
    ax.set_yscale(scale)
    ax.set_ylabel("# of events")
    ax.set_title(feature)
    ax.legend(["Electrons", "Muons"], loc='best')
    fig.savefig(f'{figdir}feature_{feature}{tag}.png')


def plot_ROC(dataset, figdir):
    # Plot the ROC (Receiver Operating Characteristic) curve

    n_points = 101
    # Make a list of threshold values not equally distant (follow a power law to have more points close to 0 and 1
    list_threshold = np.concatenate((0.5 * np.power(np.linspace(0,1,n_points//2+1),4), 1 - 0.5 * np.power(np.linspace(1,0,n_points//2+1),4)), axis=0)
    list_threshold = np.delete(list_threshold, n_points//2+1)

    true_negative = np.zeros(n_points)
    false_positive = np.zeros(n_points)
    false_negative = np.zeros(n_points)
    true_positive = np.zeros(n_points)
    dataset['predict_label_temp'] = dataset['label']

    for i in range(n_points):
        dataset['predict_label_temp'] = (dataset['prediction'] >= list_threshold[i]).astype(int)
        confusion_matrix_temp = tf.math.confusion_matrix(dataset['label'], dataset['predict_label_temp'], num_classes=2)

        true_negative[i], false_positive[i] = confusion_matrix_temp[0].numpy()
        false_negative[i], true_positive[i] = confusion_matrix_temp[1].numpy()

    TPR = true_positive / (true_positive + false_negative)
    TNR = true_negative / (true_negative + false_positive)
    FNR = 1 - TPR
    FPR = 1 - TNR

    purity = true_positive / (true_positive + false_positive)
    significance = true_positive / np.sqrt(true_positive + false_positive)
    max_significance_idx = np.nanargmax(significance)

    print("\n Max significance at threshold = ", list_threshold[max_significance_idx])
    print("\n Accuracy at this threshold = ", TPR[max_significance_idx], "\n")

    fig,ax = plt.subplots(1,1)
    ax.plot(TPR, TNR, '-b')
    ax.set_xlabel("Signal efficiency (true positive rate)")
    ax.set_ylabel("Background rejection (true negative rate)")
    ax.set_title("ROC curve")
    fig.savefig(f'{figdir}roc.png')

    fig,ax = plt.subplots(1,1)
    ax.plot(list_threshold, TPR, '-k')
    ax.plot(list_threshold, TNR, '-b')
    ax.plot(list_threshold, purity, '-g')

    ax2 = ax.twinx()
    ax2.plot(list_threshold, significance, '-r')

    ax.set_xlabel("Cut threshold value")
    ax.set_ylabel("Efficiency / Purity")
    ax2.set_ylabel("Significance")
    ax.legend(["Signal efficiency", "Background rejection", "Signal purity"], loc="lower left")
    ax2.legend(["Significance = S/sqrt(S+B)"], loc="lower right")
    fig.savefig(f'{figdir}eff.png')

    return dataset


def plot_history(history_file, result, figdir):
    # Plot loss history
    with open(history_file, 'r') as json_file:
        history = json.load(json_file)
    fig,ax = plt.subplots(1,1)
    ax.plot(history["loss"])
    ax.plot(history["val_loss"])
    ax.plot(len(history["loss"]), result[0], marker='o', linestyle='None')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Binary cross entropy loss")
    ax.legend(["Train", "Validation","Test"], loc='best')
    fig.savefig(f'{figdir}history.png')


def make_arguments():
    parser = argparse.ArgumentParser(description='Train a NN PID model')
    parser.add_argument("--data-dir", "-d", type=str, default="/exp/mu2e/data/users/mmackenz/trkpid/data/", help="Directory with data files")
    parser.add_argument("--signal-file", "-s", type=str, default="nts.flate.root", help="Signal data file")
    parser.add_argument("--background-file", "-b", type=str, default="nts.flatmu.root", help="Background data file")
    parser.add_argument("--version", "-V", type=int, default=0, help="Training version")
    parser.add_argument("--skip-import", "-I", action='store_true', help="Skip importing of data into csv file")
    parser.add_argument("--skip-train", "-T", action='store_true', help="Skip training of the model")
    parser.add_argument("--skip-export", "-E", action='store_true', help="Skip exporting the model to Onnx")
    parser.add_argument("--n-train", "-n", type=int, default=70000, help="Number of events to use in training (min with fraction)")
    parser.add_argument("--frac-train", "-f", type=float, default=0.7, help="Fraction of events to use in training (min with N(train))")
    parser.add_argument("--skip-plot", "-P", action='store_true', help="Skip plotting of results")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = make_arguments()
    data_dir = args.data_dir

    version = args.version # Version of the training features to use
    figdir  = f'figures/v{version}/'
    sig_csv_name = "array_test_signal.csv"
    bkg_csv_name = "array_test_background.csv"
    if not args.skip_import:
        make_dataset('e' , data_dir + args.signal_file    , sig_csv_name)
        make_dataset('mu', data_dir + args.background_file, bkg_csv_name)

    df_sig = pd.read_csv(sig_csv_name, index_col=0)
    df_bkg = pd.read_csv(bkg_csv_name, index_col=0)
    df_sig = add_variables(df_sig)
    df_bkg = add_variables(df_bkg)

    # Make input features
    if version == 0:
        features = ['nActiveFrac', 'nNullFrac', 'fitcon', 'dtdz_ratio']
    else:
        raise ValueError(f'Unknown training verion value {version}')
    print(f'>>> Using input features {features}')
    n_features = len(features)

    df_sig_feature = df_sig[features+['label']].copy()
    df_bkg_feature = df_bkg[features+['label']].copy()

    n_sig = len(df_sig_feature)
    n_bkg = len(df_bkg_feature)

    # Train with equal amounts of signal and background
    frac_train   = args.frac_train
    max_train    = int(2*int(min(int(2.*frac_train*min(n_sig,n_bkg)), args.n_train)/2))
    half_train   = int(max_train/2)
    df_sig_train = df_sig_feature.iloc[:half_train,:]
    df_bkg_train = df_bkg_feature.iloc[:half_train,:]
    df_sig_test  = df_sig_feature.iloc[half_train:,:]
    df_bkg_test  = df_bkg_feature.iloc[half_train:,:]
    df_train     = pd.concat([df_sig_train, df_bkg_train]).sample(frac=1, random_state=90) # Shuffle the inputs
    df_test      = pd.concat([df_sig_test , df_bkg_test ]).sample(frac=1, random_state=90)

    print(f'>>> Performing training with {max_train} from the input {n_sig+n_bkg} events')

    if not args.skip_train:
        PID_model = train_model(df_train)
    else:   # Use an already trained model saved in keras format
        PID_model = tf.keras.models.load_model("TrkOnlyPID_model.keras")
        with open("train_history.json",'r') as history_file:    # open file containing the training history to plot later
            train_history = json.load(history_file)
        n_epochs = len(train_history['loss'])

    # export model in onnx format to be able to use it with SOFIE inference code ; manually enter name and shape of input and output for SOFIE
    PID_model.output_names = ['output']
    if not args.skip_export:
        print('>>> Loading ONXX packages')
        import tf2onnx
        import onnx
        print('>>> Exporting to ONXX')
        onnx_signature = [tf.TensorSpec(input.shape, dtype=input.dtype, name=input.name) for input in PID_model.inputs]
        onnx_model, _ = tf2onnx.convert.from_keras(PID_model, input_signature=onnx_signature)
        onnx.save(onnx_model, "TrkOnlyPID.onnx")

    df_train,results_train,confusion_matrix_train = make_results(PID_model, df_train, "train", 0.5)
    df_test ,results_test ,confusion_matrix_test  = make_results(PID_model, df_test , "test" , 0.5)

    df_train_e  = df_train[df_train["label"] == 1]
    df_train_mu = df_train[df_train["label"] == 0]
    df_test_e   = df_test [df_test ["label"] == 1]
    df_test_mu  = df_test [df_test ["label"] == 0]

    # Create plots
    if not args.skip_plot:
        path = Path(figdir)
        path.mkdir(parents=True, exist_ok=True)
        plot_dataset(sig_csv_name,"Electrons", figdir)
        for feature in features:
            plot_feature(df_sig_test, df_bkg_test, feature, figdir)
        plot_feature(df_train_e, df_train_mu, "prediction", figdir, 'log', '_train')
        plot_feature(df_test_e , df_test_mu , "prediction", figdir, 'log')
        df_test = plot_ROC(df_test, figdir)
        plot_history("train_history.json", results_test, figdir)
        plot_model(PID_model, figdir)
