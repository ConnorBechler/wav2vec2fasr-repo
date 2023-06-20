from math import ceil

chunks = [[0, 1246], [2407, 22004], [25211, 38052], [39342, 44762], [48019, 81224], [82345, 87669], 
          [91116, 107550], [110387, 120828], [122398, 127972], [129352, 129373], [131077, 131640]]

nchunks = []
max_chunk = 10000
min_chunk = 100
stride = 1000
aud = list(range(131640))
for chunk in chunks:
        start, stop = chunk[0], chunk[1]
        diff = stop-start
        if diff > max_chunk:
            num_chunks = ceil(diff/max_chunk)
            step = round(diff/num_chunks)
            nchunks.append([start, start+step+stride, (0, stride)])
            for x in range(1, num_chunks):
                nchunks.append([(start+x*step)-stride, (start+(x+1)*step)+stride, (stride, stride)])
            nchunks.append([(start+(num_chunks)*step)-stride, stop, (stride, 0)])
        elif diff > min_chunk:
            nchunks.append([start, stop, (0, 0)])
print(chunks)
print(nchunks)