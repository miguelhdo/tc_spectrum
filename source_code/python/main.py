import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path2reservation', help="Full path to reservation", required=True)
    parser.add_argument('-r', '--reservation', help="Folder of the batch (e.g. RESERVATION-XXXXXX", required=True)
    parser.add_argument('-t', '--batch_type', help="Indicates if it is either a batch (b) or a freeplay (fp)", required=True)
    parser.add_argument('-th', '--incumbent_th', help="Threshold to determine if a prediction is true for Act Incumbent", required=True)

    opts = parser.parse_args()
    reservation = opts.reservation
    path_to_reservation = opts.path2reservation
    inc_th = float(opts.incumbent_th)
    type_batch = opts.batch_type
    print(reservation, path_to_reservation)
    #tr_log_file_tech_rec(path_to_reservation, reservation, type_batch, inc_th)

if __name__ == "__main__":
    main()