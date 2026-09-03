# Builds the reference-table generator against a local ipole checkout (see symphony_table.c).
IPOLE_DIR ?= $(HOME)/local_scripts/ipole
SYM = $(IPOLE_DIR)/src/symphony
SRCS = symphony_table.c $(SYM)/fits.c $(SYM)/maxwell_juettner_fits.c $(SYM)/power_law_fits.c $(SYM)/kappa_fits.c $(SYM)/params.c
CFLAGS = -O2 -I$(SYM) -I$(IPOLE_DIR)/src -DDEBUG=0
LDLIBS = -lgsl -lgslcblas -lm

tables: symphony_table
	./symphony_table

symphony_table: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDLIBS)

clean:
	rm -f symphony_table
