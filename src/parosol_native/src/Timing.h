/*
 * ParOSol: a parallel FE solver for trabecular bone modeling
 * Copyright (C) 2011, Cyril Flaig
 * Adapted to use MPI_Wtime by Fabian Keller, 2017
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef TIMING_H
#define TIMING_H

#include <map>
#include <string>
#include <mpi.h>

#define COUTTIME(msg) \
	"avg: " << msg.avg << " s, min: " <<  msg.min << " s, max: " << msg.max << " s, tot: " << msg.tot << "s"

struct t_timing
{
	double min;     // Time of quickest process
	double avg;     // Avg time of all processes
    double max;     // Time of slowest process
    double tot;     // Sum of time of all processes
    double num;     // Number of processes
};

/*! A simple class to time some parts of the code. This object can start and stop different timers. 
 * 
 * \verbatim
t.start("a"); bla();t.stop("a"); foo(); t.Restart("a"); foobar(); t.Stop("a");
t_timing t_elapsed = t.ElapsedTime("a");
\endverbatim
 */

class Timer
{
	public:
	  	/** 
		 * @brief Constructor
		 * 
		 * @param a_comm MPI Communicator
		 */
		Timer(MPI_Comm a_comm):_comm(a_comm) {
		    MPI_Comm_size(a_comm, &_comm_size);
		}

        /**
         * @brief Starts a named timer. It uses a Barrier to start the timer on all cpu at the same time.
         * 
         * @param timer name of the timer
         */
		void Start(std::string timer)
		{
            MPI_Barrier(_comm);
            _start[timer] = MPI_Wtime();
            _elapsed[timer] = 0;
		}

        /**
         * @brief Restarts a (stopped) timer. 
         * 
         * @param timer name of the timer
         */
		void Restart(std::string timer)
		{
            if (_elapsed.find(timer) == _elapsed.end()) 
            {
                // Timer not yet started
                _elapsed[timer] = 0; 
            }

            _start[timer] = MPI_Wtime();
		}

        /**
         * @brief Stops a (running) timer. 
         * 
         * @param timer name of the timer
         */
		void Stop(std::string timer)
		{
            if (_elapsed.find(timer) == _elapsed.end()) 
            {
                // Timer not yet started
                _elapsed[timer] = 0;
            }
            else
            {
                // Timer started
                _elapsed[timer] += (MPI_Wtime() - _start[timer]);            
                _start.erase(timer);
            }
		}

        /**
         * @brief Computes the time that is elapsed between starting and stopping a timer. 
         * 
         * It uses MPI_Allreduce to compute the timings
         * 
         * @param timer name of the timer
         * @return t_timing struct with the timings.
         */
		t_timing ElapsedTime(std::string timer)
		{
            double min = 0, max = 0, tot = 0;

            if (_elapsed.find(timer) == _elapsed.end()) 
            {
                // Timer not yet stopped
                Stop(timer);
            }

            double res = _elapsed[timer];
            MPI_Allreduce(&res, &max, 1, MPI_DOUBLE, MPI_MAX, _comm);
            MPI_Allreduce(&res, &min, 1, MPI_DOUBLE, MPI_MIN, _comm);
            MPI_Allreduce(&res, &tot, 1, MPI_DOUBLE, MPI_SUM, _comm);
            
            t_timing t;
			t.min = min;
            t.max = max;
            t.tot = tot;
            t.avg = tot / _comm_size; 
            t.num = _comm_size;
            return t;
		}

    private:
        std::map<std::string, double> _start;
        std::map<std::string, double> _elapsed;
        MPI_Comm _comm;
        int _comm_size;
};
#endif
