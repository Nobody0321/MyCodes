N, M, K, Q = list(map(int, input().split()))
q_times = list(map(int, input().split()))
q_customers = list(map(int, input().split()))
serving = q_times[:N]
wait_inline = q_times[N:N*M]

