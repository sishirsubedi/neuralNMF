import logging

def setlogger(sample,sample_dir):
    log_file = sample_dir+'results/'+sample+'_model.log'
    logging.basicConfig(filename=log_file,
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')