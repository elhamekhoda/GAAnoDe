import pandas as pd
import numpy as np
import os
import argparse

minmass = 3.3
maxmass = 3.7

def read_input_dfs(features, data_type):
    # Read from data
    mj1mj2 = np.array(features[['mj1','mj2']])
    tau21 = np.array(features[['tau2j1','tau2j2']])/(1e-5+np.array(features[['tau1j1','tau1j2']]))
    tau32 = np.array(features[['tau3j1','tau3j2']])/(1e-5+np.array(features[['tau2j1','tau2j2']]))
    
    # Sorting of mj1 and mj2:
    # Identifies which column has the minimum of mj1 and mj2, and sorts it so the new array mjmin contains the 
    # mj with the smallest energy, and mjmax is the one with the biggest.
    mjmin = mj1mj2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)] 
    mjmax = mj1mj2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    # Sort the taus accoring to the masses
    tau21min = tau21[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    tau21max = tau21[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]
    tau32min = tau32[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    tau32max = tau32[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    # Calculate mjj and collect the features into a dataset, plus mark signal/bg with 1/0
    pjj = (np.array(features[['pxj1','pyj1','pzj1']])+np.array(features[['pxj2','pyj2','pzj2']]))
    Ejj = np.sqrt(np.sum(np.array(features[['pxj1','pyj1','pzj1','mj1']])**2, axis=1))\
    +np.sqrt(np.sum(np.array(features[['pxj2','pyj2','pzj2','mj2']])**2, axis=1))
    mjj = np.sqrt(Ejj**2-np.sum(pjj**2, axis=1))

    if data_type == "bkg":
        labels = np.zeros(len(mjj))
    elif data_type == "sig":
        labels = np.ones(len(mjj))


    # format of dataset: mjj (TeV), mjmin (TeV), mjmax-mjmin (TeV), tau21(mjmin), tau21 (mjmax), tau32(mjmin), tau32 (mjmax), sigorbg label
    dataset = np.dstack((mjj/1000, mjmin/1000, (mjmax-mjmin)/1000, tau21min, tau21max, tau32min, tau32max, labels))[0]

    return dataset

def shuffle_dataset(data):
    
    indices = np.array(range(len(data))).astype('int')
    np.random.shuffle(indices)
    data = data[indices]

    return data

def filter_dataset (dataset, region):

    if region == "sr":
        mask = (dataset[:,0]>minmass) & (dataset[:,0]<maxmass)
    elif region == "sb":
        mask = (dataset[:,0]<=minmass) | (dataset[:,0]>=maxmass)  ## sb_mask = ~sr_mask
    return dataset[mask]

def create_synth_data(bkg, sig, n_sig):
    synth_data_all = np.concatenate((bkg, sig[:n_sig]))
    synth_data_all = shuffle_dataset(synth_data_all)

    sr_synth_data = filter_dataset(synth_data_all, "sr")
    sb_synth_data = filter_dataset(synth_data_all, "sb")

    sb_train_size = 800000
    sb_synth_data_train = sb_synth_data[:sb_train_size]
    sb_synth_data_test = sb_synth_data[sb_train_size:]
    return sb_synth_data_train, sb_synth_data_test, sr_synth_data[:122100]

def create_dataframe(npy_arr, out_filename, out_path):
    outfile_keys = ["mjj", "mjmin", "mjdiff", "tau21min", "tau21max", "tau32min", "tau32max", "sblabel"]
    df = pd.DataFrame(npy_arr, columns = outfile_keys)
    df.to_hdf(out_path+out_filename+'.h5', key='df', mode='w')
    print("Writing file: "+out_path+out_filename+'.h5')
    print("out_filename: shape = ", npy_arr.shape)




def main():

    parser = argparse.ArgumentParser(
        description=("Prepare LHCO dataset."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datapath", type=str, default="./",
                        help="path to the folder where you downloaded the LHCO raw files")
    parser.add_argument("--outdir", type=str, default="preprocessed_data/",
                        help="output directory")
    parser.add_argument("--S_over_B", type=float, default=-1,
                        help="Signal over background ratio in the signal region.")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed for the mixing")
    args = parser.parse_args()


    print("")
    print("     \033[37;44;1m DATA PREPARATION: TRAINING and TESTING DATA FOR GAN / VAE  \033[0m")
    print("-----------------------------------------------------------------------")
    print("Path to the raw LHCO files   :", args.datapath )
    print("Output directory path        :", args.outdir )
    print("S_over_B                     :", args.S_over_B)
    print("Random seed                  :", args.seed)
    print("-----------------------------------------------------------------------")

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # background + signal file: total 1.1 M events (1 M bkg and 100k 2-prong signal events)
    features_all = pd.read_hdf("/global/cscratch1/sd/elham/lhco2020_data/cGAN_data/events_anomalydetection_v2.features.h5")

    # the data containing only the 3-prong signal
    features_sig3prong = pd.read_hdf("/global/cscratch1/sd/elham/lhco2020_data/cGAN_data/events_anomalydetection_Z_XY_qqq.features.h5")  #/pscratch/sd/e/elham/lhco_data

    # additionally produced bkg
    features_extrabkg = pd.read_hdf("/global/cscratch1/sd/elham/lhco2020_data/cGAN_data/events_anomalydetection_qcd_extra_inneronly_features.h5")

    ## to be split among the different sets 
    features_extrabkg1 = features_extrabkg[:312858]

    ## to be used to enhance the evalaution
    features_extrabkg2 = features_extrabkg[312858:]

    features_sig2prong = features_all[features_all['label']==1]
    features_bg = features_all[features_all['label']==0]

    # define the datasets 
    dataset_bg = read_input_dfs(features_bg, "bkg")
    dataset_sig2prong = read_input_dfs(features_sig2prong, "sig")
    dataset_sig3prong = read_input_dfs(features_sig3prong, "sig")
    dataset_extrabkg1 = read_input_dfs(features_extrabkg1, "bkg")
    dataset_extrabkg2 = read_input_dfs(features_extrabkg2, "bkg")

    np.random.seed(args.seed) # Set the random seed so we get a deterministic result

    if args.seed!=1:
        # np.random.shuffle(dataset_sig)
        np.random.shuffle(dataset_sig3prong)

    if args.S_over_B==-1:
        n_sig = 1000
    else:
        n_sig = int(args.S_over_B*1000/0.006361658645922605)

    # format of data_all: mjj (TeV), mjmin (TeV), mjmax-mjmin (TeV), tau21(mjmin), tau21 (mjmax), tau32(mjmin), tau32 (mjmax), sigorbg label

    print("\033[34;1mSignal Region: {} - {} TeV \033[0m".format(minmass, maxmass))
    print("-----------------------------------------------------------------------")


    sb_synth_data_train_sig2prong, sb_synth_data_test_sig2prong, sr_synth_data_sig2prong = create_synth_data(dataset_bg, dataset_sig2prong, n_sig)
    sb_synth_data_train_sig3prong, sb_synth_data_test_sig3prong, sr_synth_data_sig3prong = create_synth_data(dataset_bg, dataset_sig3prong, n_sig)

    dataset_extrabkg1 = shuffle_dataset(dataset_extrabkg1)
    dataset_extrabkg2 = shuffle_dataset(dataset_extrabkg2)

    sr_extrabkg1 = filter_dataset(dataset_extrabkg1, "sr")
    sr_extrabkg2 = filter_dataset(dataset_extrabkg2, "sr")

    # print ("Shape of the SR training set: ", sr_train.shape)

    # lets use rest of the signal events (not used for created the synthetic data) for the classifier
    sr_extrasig_2prong = dataset_sig2prong[n_sig:]
    sr_sig2prong = filter_dataset(sr_extrasig_2prong, "sr")

    sr_extrasig_3prong = dataset_sig3prong[n_sig:]
    sr_sig3prong = filter_dataset(sr_extrasig_3prong, "sr")

    ## splitting extra signal into train and test set
    n_sig_test = 20000

    # 2-prong signal train and test set
    sr_sig2prong_test = sr_sig2prong[:n_sig_test]
    sr_sig2prong_train = sr_sig2prong[n_sig_test:]
    # 3-prong signal train and test set
    sr_sig3prong_test = sr_sig3prong[:n_sig_test]
    sr_sig3prong_train = sr_sig3prong[n_sig_test:]

    ## splitting extra bkg (1) into train, val and test set
    n_bkg_test = 40000
    sr_extrabkg_train = sr_extrabkg1[n_bkg_test:]
    sr_extrabkg1_test = sr_extrabkg1[:n_bkg_test]
    # sr_extrabkg2 has 300k events. Combine sr_extrabkg_test with it to create a test set of 340k bkg events
    # putting together with signal to create the synthetic dataset
    sr_sig2prong_classifier_test = np.concatenate((sr_extrabkg1_test, sr_extrabkg2, sr_sig2prong_test) , axis=0)
    sr_sig2prong_classifier_test = shuffle_dataset(sr_sig2prong_classifier_test)
    sr_sig3prong_classifier_test = np.concatenate((sr_extrabkg1_test, sr_extrabkg2, sr_sig3prong_test) , axis=0)
    sr_sig3prong_classifier_test = shuffle_dataset(sr_sig3prong_classifier_test)

    create_dataframe(sb_synth_data_train_sig2prong, 'sb_train_6var', args.outdir)
    create_dataframe(sb_synth_data_test_sig2prong, 'sb_test_6var', args.outdir)
    create_dataframe(sr_extrabkg_train, 'sr_mc_train_6var', args.outdir)
    create_dataframe(sr_synth_data_sig2prong, 'sr_synth_data_sig2prong_train_6var', args.outdir)
    create_dataframe(sr_synth_data_sig3prong, 'sr_synth_data_sig3prong_train_6var', args.outdir)
    create_dataframe(sr_sig2prong_classifier_test, 'sr_sig2prong_classifier_test_6var', args.outdir)
    create_dataframe(sr_sig3prong_classifier_test, 'sr_sig3prong_classifier_test_6var', args.outdir)
    create_dataframe(sr_sig2prong_train, 'sr_sig2prong_train_6var', args.outdir)
    create_dataframe(sr_sig3prong_train, 'sr_sig3prong_train_6var', args.outdir)
    

if __name__ == "__main__":
    main()