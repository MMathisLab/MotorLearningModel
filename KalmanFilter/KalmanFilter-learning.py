# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def SimulatingModel_long(sigma_m,sigma_s,sigma_p,a,distanceduringperturbation):
	''' Simulating Kalman-filter-based learning-model for a set of model parameters. This code is inefficient and stores many
	intermediary variables. For speed use the equivalent "SimulatingModel_fast" function. '''

	# Trialstructure from paper:   
	baseline=75             # number of trials
	perturbation=100        # number of trials
	washout=75              # number of trials
	experimentnumtrials=baseline+washout+perturbation

	# Defining force-field. Vanishes in baseline and washout and is given by deviation in first few perturbation trials of data (see methods)
	Disturbance=np.zeros(experimentnumtrials)
	Disturbance[baseline:baseline+perturbation]=distanceduringperturbation

	# Model definitions
	F=np.array([[a,0],[1,0]])                           # state transition matrix 
	Q=np.array([[sigma_p**2,0],[0,sigma_m**2]])         # state noise

	K=np.zeros((2,experimentnumtrials))         # Kalman Gain
	b=np.array([0, 1])                          # control matrix
	H=np.array([0, 1])                          # measurment matrix
	P=np.zeros((2,2))                           # covariance matrix - we assume that mouse is initially certain about state 

	# Model variables
	paw_position=np.zeros(experimentnumtrials)       # actual paw position
	u=np.zeros(experimentnumtrials)                  # motor control signal
	xhat=np.zeros((2,experimentnumtrials))           # internal variables = (perturbation/disturbance , hand pos.)^T
	y=np.zeros(experimentnumtrials)                  # sensed paw position
	eta_motor=sigma_m*np.random.randn(experimentnumtrials)            #noise motor
	eta_sensory=sigma_s*np.random.randn(experimentnumtrials)          #noise sensory
	eta_disturbance=sigma_p*np.random.randn(experimentnumtrials)      #internal model of disturbance noise  

	for t in range(experimentnumtrials-1):
		paw_position[t]=u[t]+eta_motor[t]+Disturbance[t]+eta_disturbance[t]		# paw position depends on motor command and disturbance!
		y[t]=paw_position[t]+eta_sensory[t]		# Sensed paw position depends on disturbed paw position and sensory noise
		if t>75: 				
			# Start learning with Kalman-filter during perturbation and washout blocks.
			# Forward model - (a-priori) prediction step
			x_apriori=np.dot(F,xhat[:,t-1])+b*u[t] #order: (perturbation , hand pos.) A priori estimate (without) sensory feedback
			# Kalman gain and uncertainty estimates
			K[:,t]=np.dot(P,H)*1./(P[0,0]+sigma_s**2+sigma_m**2)    # here P is a priori covariance (from last trial) P k-1 | k-1
			P=np.dot(np.dot(F,P),F.T)+Q					        # Forward prediction step covariance P k|k-1 = ...

			# A-posteriori estimate and uncertainty
			xhat[:,t]=x_apriori+K[:,t]*(y[t]-np.dot(H,x_apriori))
			P=np.dot(np.eye(2)-np.outer(K[:,t],H),P) 		     	  # Updating P (i.e. P k | k = ... 

			# Extracting motor command for next trial (= ideal estimate)
			u[t+1]=-xhat[0,t]

	# returns: the paw position; sensed paw position; motor command and perturbation estimate across time
	return	paw_position,y,u,xhat[0,:]

def SimulatingModel_fast(sigma_m,sigma_s,sigma_p,a,distanceduringperturbation):
    ''' Simulating Kalman-filter-based learning-model for a set of model parameters. This code is inefficient and stores many
	intermediary variables. For speed use the equivalent "SimulatingModel_fast" function. '''
    
	#to be done
	return 0



if __name__ in '__main__':
	import pickle

	## Loading experimental and fit data
	with open("Fig2D_data.pickle","rb") as f:
		data,data_errors,sigma_m,sigma_s,sigma_p,a,distanceduringperturbation=pickle.load(f) 

	numrealizations=10      #number of realization of (stochastic) model

	# plotting variables:
	alphavalue=.05
	lowerbound=-.125
	upperbound=.075
	lwidth=1

	plt.figure(figsize=(6, 5))    
	ax = plt.subplot(111)    
	ax.spines["top"].set_visible(False)    
	ax.spines["right"].set_visible(False)    

	# Run model for fitted parameters - gives one random iteration
	for jj in range(numrealizations):
		paw,y,u,xhat=SimulatingModel_long(sigma_m,sigma_s,sigma_p,a,distanceduringperturbation)
		plt.plot(np.arange(250)+1,paw,'--',color='red',alpha=alphavalue,lw=lwidth)

		if jj>0: # accumulating paw position for averaging over realizations (similar to multiple mice/sessions)
		    P+=paw
		else:
		    P=paw

	plt.plot(np.arange(250)+1,data,'.-',color='black',ms=5,alpha=.5)
	plt.fill_between(np.arange(250)+1,data-data_errors,data+data_errors,facecolor='black',edgecolor='black',alpha=.15)

	plt.plot(np.arange(250)+1,P*1./numrealizations,'.-',color='red',ms=5,alpha=.5)
	plt.yticks([-.1,-.05,0,.05],[-2,-1,0,1])
	plt.xlim([0,251])
	plt.ylabel("Perpendicular displacement")
	plt.xlabel("Trial ")
	plt.ylim(lowerbound,upperbound)

	plt.savefig("Figure2D_realization.pdf")


	 
