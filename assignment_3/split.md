## TASK DISTRIBUTION

# Q1 (HYDRA, Nabila)
> Figure out how to get to global optimum on the provided data sets

# Q2  (HYDRA, Nabila)
> Experiment with cooling schedules

# Q3 (HYDRA, Nabila)
> What is the effect of the MCMC length

# Q4 
> [List based Simulated annealing](https://www.hindawi.com/journals/cin/2016/1712630/) (Sophie)

# Q5 (Amir)
>  [Adaptive cooling](https://arxiv.org/pdf/2002.06124.pdf)

# Q6 (Amir)
> [Stopping criteria](https://sci-hub.yncjkj.com/10.1109/iccd.1988.25760) (Instituational access means nothing anymore)

> [We did this](https://doi.org/10.1016/0375-9601(87)90796-1)

# Q7 Simulated annealing alternatives

> [Simply fast annealing](https://pdf.sciencedirectassets.com/271541/1-s2.0-S0375960100X14703/1-s2.0-0375960187907961/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJ3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCMlkaLi9O6VF7NHDDaHLkuCb%2F2uxEq11U%2BW%2FgBA%2FmLbwIhAKUhEmvu85ahtvvYxo1XfDm0dPYobZ5Zovn81bOARRT5KrMFCBYQBRoMMDU5MDAzNTQ2ODY1IgwFJ3iIL9e%2F1JxinAMqkAV3B4WDosehJN7l%2FD5b0YQiXyEyfukcoYLgWSQ3fjJSngyUxH6HU1R1k78YTamOg5epM532vNOB4jKb3DP5w90cUSa7LMVeVtrTwpTufeuDlOMl%2F3pHEiihc%2Bzq4FMs4%2BrvEmzPNRQBqTVQ0rfneA0EF37hnKHf9B6shPJdlsWjxJkk3GWUpkKLHuxK18YaWY0%2BCe%2ByyYh%2BWk44VvbskBNPy0sOaZfE8t6FCfJgOcm6CPefFWGav0Hj1d7hRhz9MgXJH4z6HkSMhO%2F7J3st2w0wBWEMbordigj%2Brh8kx%2F7frxxMREXrpIhdy1CigrbKNkQGOsWODNGnPEgz%2FaInxbK4jJvpcgmNPnbSNWcL%2BVtFeM1yDqYHNVRTyOoQ5pXg60Dxsm1wMEpMOW67Q%2FvgrD8NKnfisuWCLqEaBV010KoYE0hvZm1fp8YcJ%2Br%2FB%2B3kQdzhXNpsw6vG3v9pXacGgeo3sdNoJVn76ykkXjzTt7DkcLN%2BtvBK0ZA1bevC9PjeBvRehdmw1FWlsRim4RbNMG4DlmKFSCrRDT%2BVQmTol4XaZXjsPtIUC7GgiiNcgiKxu2BXbS55QpjwLwQ27ZQsJP3nMouTizo2pscePM73LEzsL508gLpRnOCKz558%2FMl9wfjI%2FQOwS859LoD%2BstMIMkEeH4RU3Co8tRAfqaH92RVyK5a3p97W37tPSKg1Cwyd5Wj59zgYNVt6LGg5B03iugBYJHPjwHgByKsT2l12zHzpt8shkrprewNnf5quOj60d9Nauqjjn%2Fv70D2pndDpZL3mcivRUtQwpMJpAscsfIOyO78QO0mlsaJg1sVSn7uJQx1YL%2FU68wH57Vrloy2cXyJb0wblrF3X1AVLVpvKMwJ6TjDpjtyrBjqwAb50UYfDvq8LERD6yRoQnow32Xxxq%2B60IsojDUC%2BJJczqWX57xD17Mv%2F6rSbx18Z5r569Le3clGNqRUkV%2BiIac1ulmf%2BRsMoTPmLq95htvgFnGG1e6vQENAyHStAdb5AiJIEPPAiNwmXjV5e2g7YFzv4nyKFny7FwWbBdW1W92V%2BNsaJ4UMazUXKRxioNSXRo0a8B89KBRLhavep7tlVwA6trK3QgAc6DZJshTGBN9lo&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231211T143340Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY2W2PZTZE%2F20231211%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=7459be95e4708960b454bf409e76225d6e3d2692dfdf94090f00c0b1630079d6&hash=3c118b1f0bf48b8f36ce70b671b858ab1692b917b7f39f3486850970e49aef6a&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=0375960187907961&tid=spdf-42dcec6c-5dd7-4198-9ce9-5bca3b2c38cf&sid=b1b6e57a11fbb340b629e2780b577504eb3cgxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=14095f5d5405505703060b&rr=833e71acedb10b38&cc=nl)
> [Dual annealing, combining classical and fast annealing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html)

# Visualizations
- convergence plots (best case)
- imshow for cooling rate vs chain length
    - average or best (SAVED DATA FOR BOTH)
- Compare FSA and SA for different cooling rate *using the same chain length found in previous step*


# Stats
| method  | Small map | Medium map | Large map|
|-------------- | -------------- | -------------- | ----|
|-------------- | Mean/Best/Conf| Mean/Best/Conf | Mean/Best/Conf
|-------------- | -------------- | -------------- | ----|
|SA| X/Y/+-Z|X/Y/+-Z | X/Y/+-Z|
|FSA| -------------- | -------------- | ----|
|LSA| -------------- | -------------- | ----|




