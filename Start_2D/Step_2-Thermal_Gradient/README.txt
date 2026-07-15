This folder holds the code and results obtained when learning the CA upon thermal gradients and directional grain growth.

The level of cooling rate also determines if the program can finish grain formation within the specified steps or not. High undercooling rate ( >=2 ) can easily finish before the steps runout.
Low cooling rate (~1) will prohibit grain formation withing the limited steps and teh final result will be left with molten material remaining.

The image attached with this as sample is formed for 20 random nuclie distributed throughout the sample piece.

why V1 and V2: 
V1 - has predefined number of steps so the simulation will stop eventually anyway. Neither checks the condition whether any liquid remains after simulation ends, nor checks if everything has already solidified and still keeps running the code.
V2 - has a "while" loop instead of a "for" loop which ensures the code running till all liquid solidifies. The boundaries create an issue where they are never dealt with and hence always stay liquid in V1. Thus I explicitely define them -1 so they are not 0(liquid) anymore and are skipped from the checking in the code.
