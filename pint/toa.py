import re, sys, os, cPickle, numpy, gzip
from . import utils
from . import observatories as obsmod
from . import erfautils
import spice
import astropy.time as time
import astropy.table as table
import astropy.units as u
from astropy.time.core import SECS_PER_DAY
from spiceutils import objPosVel, load_kernels
from pint import pintdir, J2000, J2000ld
from astropy import log

toa_commands = ("DITHER", "EFAC", "EMAX", "EMAP", "EMIN", "EQUAD", "FMAX",
                "FMIN", "INCLUDE", "INFO", "JUMP", "MODE", "NOSKIP", "PHA1",
                "PHA2", "PHASE", "SEARCH", "SIGMA", "SIM", "SKIP", "TIME",
                "TRACK", "ZAWGT", "FORMAT", "END")

observatories = obsmod.read_observatories()
iers_a_file = None
iers_a = None

def get_TOAs(timfile, ephem="DE421", planets=False):
    """Convenience function to load and prepare TOAs for PINT use.

    Loads TOAs from a '.tim' file, applies clock corrections, computes
    key values (like TDB), computes the observatory position and velocity
    vectors, and pickles the file for later use.
    """
    t = TOAs(timfile)
    log.info("Applying clock corrections.")
    t.apply_clock_corrections()
    if 'tdb' not in t.table.colnames:
        log.info("Getting IERS params and computing TDBs.")
        t.compute_TDBs()
    if 'pvs' not in t.table.colnames:
        log.info("Computing observatory positions and velocities.")
        t.compute_posvels(ephem, planets)
    if not (os.path.isfile(timfile+".pickle") or
            os.path.isfile(timfile+".pickle.gz")):
        log.info("Pickling TOAs.")
        t.pickle()
    return t

def toa_format(line, fmt="Unknown"):
    """Determine the type of a TOA line.

    Identifies a TOA line as one of the following types:
    Comment, Command, Blank, Tempo2, Princeton, ITOA, Parkes, Unknown.
    """
    if re.match(r"[0-9a-z@] ", line):
        return "Princeton"
    elif line[0] == 'C' or line[0] == '#':
        return "Comment"
    elif line.startswith(toa_commands):
        return "Command"
    elif re.match(r"^\s+$", line):
        return "Blank"
    elif len(line) > 80 or fmt == "Tempo2":
        return "Tempo2"
    elif re.match(r"  ", line) and len(line) > 41 and line[41] == '.':
        return "Parkes"
    elif re.match(r"\S\S", line) and len(line) > 14 and line[14] == '.':
        # FIXME: This needs to be better
        return "ITOA"
    else:
        return "Unknown"

def get_obs(obscode):
    """Search for an observatory by obscode in the PINT observatories.txt file."""
    if obscode in ['@', 'SSB', 'BARY', 'BARYCENTER']:
        return "Barycenter"
    for name in observatories:
        if obscode in observatories[name].aliases:
            return name
    raise ValueError("cannot identify observatory '%s'!" % obscode)

def parse_TOA_line(line, fmt="Unknown"):
    """Parse a one-line ASCII time-of-arrival.

    Return an MJD tuple and a dictionary of other TOA information.
    The format can be one of: Comment, Command, Blank, Tempo2,
    Princeton, ITOA, Parkes, or Unknown.
    """
    MJD = None
    fmt = toa_format(line, fmt)
    d = dict(format=fmt)
    if fmt == "Princeton":
        fields = line.split()
        d["obs"] = get_obs(line[0].upper())
        d["freq"] = float(fields[1])
        d["error"] = float(fields[3])
        ii, ff = fields[2].split('.')
        MJD = (int(ii), float("0."+ff))
        try:
            d["ddm"] = float(fields[4])
        except IndexError:
            d["ddm"] = 0.0
    elif fmt == "Tempo2":
        # This could use more error catching...
        fields = line.split()
        d["name"] = fields[0]
        d["freq"] = float(fields[1])
        ii, ff = fields[2].split('.')
        MJD = (int(ii), float("0."+ff))
        d["error"] = float(fields[3])
        d["obs"] = get_obs(fields[4].upper())
        # All the rest should be flags
        flags = fields[5:]
        for i in range(0, len(flags), 2):
            k, v = flags[i].lstrip('-'), flags[i+1]
            try:  # Convert what we can to floats and ints
                d[k] = int(v)
            except ValueError:
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
    elif fmt == "Command":
        d[fmt] = line.split()
    elif fmt == "Parkes" or fmt == "ITOA":
        raise RuntimeError(
            "TOA format '%s' not implemented yet" % fmt)
    return MJD, d

class TOA(object):
    """A time of arrival (TOA) class.

        MJD will be stored in astropy.time.Time format, and can be
            passed as a double (not recommended), a string, a
            tuple of component parts (day and fraction of day).
        error is the TOA uncertainty in microseconds
        obs is the observatory name as defined in XXX
        freq is the observatory-centric frequency in MHz
        freq
        other keyword/value pairs can be specified as needed

    Example:
        >>> a = TOA((54567, 0.876876876876876), 4.5, freq=1400.0,
        ...         obs="GBT", backend="GUPPI")
        >>> print a
        54567.876876876876876:  4.500 us error from 'GBT' at 1400.0000 MHz {'backend': 'GUPPI'}

    What happens if IERS data is not available for the date:
        >>> a = TOA((154567, 0.876876876876876), 4.5, freq=1400.0,
        ...         obs="GBT", backend="GUPPI")
        Traceback (most recent call last):
          omitted
        IndexError: (some) times are outside of range covered by IERS table.

    """
    def __init__(self, MJD, # required
                 error=0.0, obs='bary', freq=float("inf"),
                 scale='utc', # with defaults
                 **kwargs):  # keyword args that are completely optional
        if obs not in observatories:
            raise ValueError("Unknown observatory %s" % obs)
        if obs is "Barycenter":
            self.mjd = time.Time(MJD[0], MJD[1],
                                scale='tdb', format='mjd',
                                precision=9)
        else:
            self.mjd = time.Time(MJD[0], MJD[1],
                                scale=scale, format='mjd',
                                location=observatories[obs].loc,
                                precision=9)
        self.error = error
        self.obs = obs
        self.freq = freq
        self.flags = kwargs

    def __str__(self):
        s = utils.time_to_mjd_string(self.mjd) + \
            ": %6.3f us error from '%s' at %.4f MHz " % \
            (self.error, self.obs, self.freq)
        if len(self.flags):
            s += str(self.flags)
        return s
 

class TOAs(object):
    """A class of multiple TOAs, loaded from zero or more files."""
    def __init__(self, toafile=None, usepickle=True):
        if toafile:
            if type(toafile) in [tuple, list]:
                self.filename = None
                for infile in toafile:
                    self.read_toa_file(infile, usepickle=usepickle)
            else:
                pth, ext = os.path.splitext(toafile)
                if ext == ".pickle":
                    toafile = pth
                elif ext == ".gz":
                    pth0, ext0 = os.path.splitext(pth)
                    if ext0 == ".pickle":
                        toafile = pth0
                self.read_toa_file(toafile, usepickle=usepickle)
                self.filename = toafile
        # FIXME: work with file-like objects
        else:
            self.toas = []
            self.commands = []
            self.filename = None
        if not hasattr(self, 'table'):
            # The table is grouped by observatory
            self.table = table.Table([numpy.arange(self.ntoas), self.get_mjds(),
                                      self.get_errors()*u.us, self.get_freqs()*u.MHz,
                                      self.get_obss(), self.get_flags()],
                                      names=("index", "mjd", "error", "freq",
                                              "obs", "flags"),
                                      meta = {'filename':self.filename}).group_by("obs")
            # We don't need this now that we have a table
            del(self.toas)

    def __add__(self, x):
        if type(x) in [int, float]:
            if not x:
                # Adding zero. Do nothing
                return self

    def __sub__(self, x):
        if type(x) in [int, float]:
            if not x:
                # Subtracting zero. Do nothing
                return self

    def get_freqs(self):
        """Return a numpy array of the observing frequencies in MHz for the TOAs"""
        if hasattr(self, "toas"):
            return numpy.array([t.freq for t in self.toas])
        else:
            return self.table['freq']

    def get_mjds(self):
        """Return a list of the astropy.times (UTC) of the TOAs"""
        if hasattr(self, "toas"):
            return numpy.array([t.mjd for t in self.toas])
        else:
            return numpy.array([t.mjd for t in self.table['mjd']])

    def get_errors(self):
        """Return a numpy array of the TOA errors in us"""
        if hasattr(self, "toas"):
            return numpy.array([t.error for t in self.toas])
        else:
            return self.table['error']

    def get_obss(self):
        """Return a numpy array of the observatories for each TOA"""
        if hasattr(self, "toas"):
            return numpy.array([t.obs for t in self.toas])
        else:
            return self.table['obs']

    def get_flags(self):
        """Return a numpy array of the TOA flags"""
        if hasattr(self, "toas"):
            return numpy.array([t.flags for t in self.toas])
        else:
            return self.table['flags']

    def pickle(self, filename=None):
        """Write the TOAs to a .pickle file with optional filename."""
        if filename is not None:
            cPickle.dump(self, open(filename, "wb"))
        elif self.filename is not None:
            cPickle.dump(self, gzip.open(self.filename+".pickle.gz", "wb"))
        else:
            sys.stderr.write("Warning: pickle needs a filename\n")

    def get_summary(self):
        """Return a short ASCII summary of the TOAs."""
        s = "Number of TOAs:  %d\n" % self.ntoas
        s += "Number of commands:  %d\n" % len(self.commands)
        s += "Number of observatories:  %d %s\n" % (len(self.observatories),
                                                    list(self.observatories))
        for ii, key in enumerate(self.table.groups.keys):
            grp = self.table.groups[ii]
            s += "%s TOAs (%d):\n" % (key['obs'], len(grp))
            s += "  Min error:     %.3g us\n" % numpy.min(grp['error'])
            s += "  Max error:     %.3g us\n" % numpy.max(grp['error'])
            s += "  Mean error:    %.3g us\n" % numpy.mean(grp['error'])
            s += "  Median error:  %.3g us\n" % numpy.median(grp['error'])
            s += "  Error stddev:  %.3g us\n" % numpy.std(grp['error'])
        return s

    def print_summary(self):
        """Write a summary of the TOAs to stdout."""
        print self.get_summary()

    def apply_clock_corrections(self):
        """Apply observatory clock corrections and TIME statments.

        Apply clock corrections to all the TOAs where corrections are
        available.  This routine actually changes the value of the TOA,
        although the correction is also listed as a new flag for the TOA
        called 'clkcorr' so that it can be reversed if necessary.  This
        routine also applies all 'TIME' commands and treats them exactly
        as if they were a part of the observatory clock corrections.
        """
        # First make sure that we haven't already applied clock corrections
        flags = self.table['flags']
        for tfs in flags:
            if 'clkcorr' in tfs:
                log.info("Some TOAs have 'clkcorr' flag.  Not applying new clock corrections.")
                return
        # An array of all the time corrections, one for each TOA
        corr = numpy.zeros(self.ntoas) * u.s
        times = self.table['mjd']
        for ii, key in enumerate(self.table.groups.keys):
            grp = self.table.groups[ii]
            obs = self.table.groups.keys[ii]['obs']
            loind, hiind = self.table.groups.indices[ii:ii+2]
            # First apply any TIME statements
            for jj in range(loind, hiind):
                if flags[jj].has_key('time'):
                    # TIME commands are in sec
                    corr[jj] = flags[jj]['time'] * u.s
                    times[jj] += time.TimeDelta(corr[jj])
            # These are observatory clock corrections.  Do in groups.
            if (key['obs'] in observatories):
                mjds, ccorr = obsmod.get_clock_corr_vals(key['obs'])
                tvals = numpy.array([t.mjd for t in grp['mjd']])
                if numpy.any((tvals < mjds[0]) | (tvals > mjds[-1])):
                    # FIXME: check the user sees this! should it be an exception?
                    log.error("Some TOAs are not covered by the %s clock correction"
                        +" file, treating clock corrections as constant"
                        +" past the ends." % obsname)
                gcorr = numpy.interp(tvals, mjds, ccorr) * u.us
                for jj, cc in enumerate(gcorr):
                    grp['mjd'][jj] += time.TimeDelta(cc)
                corr[loind:hiind] += gcorr
            # Now update the flags with the clock correction used
            for jj in range(loind, hiind):
                if corr[jj]:
                    flags[jj]['clkcorr'] = corr[jj]

    def compute_TDBs(self):
        """Compute and add TDB and TDB long double columns to the TOA table

        This routine creates new columns 'tdb' and 'tdbld' in a TOA table
        for TDB times, using the Observatory locations and IERS A Earth
        rotation corrections for UT1.
        """
        from astropy.utils.iers import IERS_A, IERS_A_URL
        from astropy.utils.data import download_file
        global iers_a_file, iers_a
        # First make sure that we have already applied clock corrections
        ccs = False
        for tfs in self.table['flags']:
            if 'clkcorr' in tfs: ccs = True
        if ccs is False:
            log.info("No TOAs have clock corrections.  Use .apply_clock_corrections() first.")
        # These will be the new table columns
        col_tdb = numpy.zeros_like(self.table['mjd'])
        col_tdbld = numpy.zeros(self.ntoas, dtype=numpy.longdouble)
        # Read the IERS for ut1_utc corrections
        iers_a_file = download_file(IERS_A_URL, cache=True)
        iers_a = IERS_A.open(iers_a_file)
        # Now step through in observatory groups to compute TDBs
        for ii, key in enumerate(self.table.groups.keys):
            grp = self.table.groups[ii]
            obs = self.table.groups.keys[ii]['obs']
            loind, hiind = self.table.groups.indices[ii:ii+2]
            if (key['obs'] in observatories):
                utcs = time.Time([t.isot for t in grp['mjd']],
                                format='isot', scale='utc', precision=9,
                                location=observatories[obs].loc)
                utcs.delta_ut1_utc = utcs.get_delta_ut1_utc(iers_a)
                # Now set the delta_ut1_utc for all the TOAs for later use
                for toa, dut1 in zip(grp['mjd'], utcs.delta_ut1_utc):
                    toa.delta_ut1_utc = dut1
                tdbs = utcs.tdb
            elif key['obs'] is "Barycenter":
                # copy the times to the tdb column
                tdbs = grp['mjd'].tdb
            col_tdb[loind:hiind] = numpy.asarray([t for t in tdbs])
            col_tdbld[loind:hiind] = numpy.asarray([utils.time_to_longdouble(t) for t in tdbs])
        # Now add the new columns to the table
        col_tdb = table.Column(name='tdb', data=col_tdb)
        col_tdbld = table.Column(name='tdbld', data=col_tdbld)
        self.table.add_columns([col_tdb, col_tdbld])

    def compute_posvels_old(self, ephem="DE421", planets=False):
        """Compute positions and velocities of observatory and Earth.

        Compute the positions and velocities of the observatory (wrt
        the Geocenter) and the center of the Earth (referenced to the
        SSB) for each TOA.  The JPL solar system ephemeris can be set
        using the 'ephem' parameter.  The positions and velocities are
        set with PosVel class instances which have astropy units.
        """
        # Load the appropriate JPL ephemeris
        load_kernels(ephem)
        pth = os.path.join(pintdir, "datafiles")
        ephem_file = os.path.join(pth, "%s.bsp"%ephem.lower())
        log.info("Loading %s ephemeris." % ephem_file)
        spice.furnsh(ephem_file)
        for toa in self.toas:
            xyz = observatories[toa.obs].xyz
            toa.obs_pvs = erfautils.topo_posvels(xyz, toa)
            # SPICE expects ephemeris time to be in sec past J2000 TDB
            # We need to figure out how to get the correct time...
            et = (toa.mjd.TDB - j2000_mjd) * SECS_PER_DAY
            # SSB to observatory position/velocity:
            toa.earth_pvs = objPosVel("EARTH", "SSB", et)
            toa.pvs = toa.obs_pvs + toa.earth_pvs
            # Obs to Sun PV:
            toa.obs_sun_pvs = objPosVel("SUN", "EARTH", et) - toa.obs_pvs
            if planets:
                for p in ('jupiter', 'saturn', 'venus', 'uranus'):
                    pv = objPosVel(p.upper()+" BARYCENTER",
                            "EARTH", et) - toa.obs_pvs
                    setattr(toa, 'obs_'+p+'_pvs', pv)
    
    def compute_posvels(self, ephem="DE421", planets=False):
        """Compute positions and velocities of the observatories and Earth.

        Compute the positions and velocities of the observatory (wrt
        the Geocenter) and the center of the Earth (referenced to the
        SSB) for each TOA.  The JPL solar system ephemeris can be set
        using the 'ephem' parameter.  The positions and velocities are
        set with PosVel class instances which have astropy units.
        """
        load_kernels(ephem)
        pth = os.path.join(pintdir, "datafiles")
        ephem_file = os.path.join(pth, "%s.bsp"%ephem.lower())
        log.info("Loading %s ephemeris." % ephem_file)
        spice.furnsh(ephem_file)
        self.table.meta['ephem'] = ephem
        pvs = table.Column(name='pvs', data=numpy.zeros_like(self.table['mjd']))
        obs_pvs = table.Column(name='obs_pvs', data=numpy.zeros_like(pvs))
        earth_pvs = table.Column(name='earth_pvs', data=numpy.zeros_like(pvs))
        obs_sun_pvs = table.Column(name='obs_sun_pvs', data=numpy.zeros_like(pvs))
        if planets:
            plan_pvs = {}
            for p in ('jupiter', 'saturn', 'venus', 'uranus'):
                pstr = 'obs_'+p+'_pvs'
                plan_pvs[pstr] = table.Column(name=pstr, data=numpy.zeros_like(pvs))
        # Now step through in observatory groups
        for ii, key in enumerate(self.table.groups.keys):
            grp = self.table.groups[ii]
            obs = self.table.groups.keys[ii]['obs']
            loind, hiind = self.table.groups.indices[ii:ii+2]
            xyz = observatories[obs].loc.geocentric
            if (key['obs'] in observatories):
                for jj, grprow in enumerate(grp):
                    obs_pvs[jj+loind] = erfautils.topo_posvels(xyz, grprow)
                    et = float((grprow['tdbld'] - J2000ld) * SECS_PER_DAY)
                    earth_pvs[jj+loind] = objPosVel("EARTH", "SSB", et)
                    pvs[jj+loind] = obs_pvs[jj+loind] + earth_pvs[jj+loind]
                    obs_sun_pvs[jj+loind] = objPosVel("SUN", "EARTH", et) - \
                        obs_pvs[jj+loind]
                    if planets:
                        for p in ('jupiter', 'saturn', 'venus', 'uranus'):
                            pstr = 'obs_'+p+'_pvs'
                            pv = objPosVel(p.upper()+"BARYCENTER",
                                           "EARTH", et) - obs_pvs[jj+loind]
                            plan_pvs[pstr][jj+loind] = pv
        cols_to_add = [pvs, obs_pvs, earth_pvs, obs_sun_pvs]
        if planets:
            cols_to_add += plan_pvs.values()
        self.table.add_columns([pvs, obs_pvs, earth_pvs, obs_sun_pvs])

    def read_toa_file(self, filename, process_includes=True, top=True, usepickle=True):
        """Read the given filename and return a list of TOA objects.

        Will process INCLUDEd files unless process_includes is False.
        """
        if top:
            # Read from a pickle file if available
            if usepickle and (os.path.isfile(filename+".pickle") or
                              os.path.isfile(filename+".pickle.gz")):
                ext = ".pickle.gz" if \
                  os.path.isfile(filename+".pickle.gz") else ".pickle"
                if (os.path.getmtime(filename+ext) >
                    os.path.getmtime(filename)):
                    log.info("Reading toas from '%s'...\n" % \
                             (filename+ext))
                    # Pickle file is newer, assume it is good and load it
                    if ext==".pickle.gz":
                        tmp = cPickle.load(gzip.open(filename+ext))
                    else:
                        tmp = cPickle.load(open(filename+ext))
                    self.filename = tmp.filename
                    if hasattr(tmp, 'toas'):
                        self.toas = tmp.toas
                    if hasattr(tmp, 'table'):
                        self.table = tmp.table.group_by("obs")
                    if hasattr(tmp, 'ntoas'):
                        self.ntoas = tmp.ntoas
                    self.commands = tmp.commands
                    self.observatories = tmp.observatories
                    return
            self.ntoas = 0
            self.toas = []
            self.commands = []
            self.cdict = {"EFAC": 1.0, "EQUAD": 0.0,
                          "EMIN": 0.0, "EMAX": 1e100,
                          "FMIN": 0.0, "FMAX": 1e100,
                          "INFO": None, "SKIP": False,
                          "TIME": 0.0, "PHASE": 0,
                          "PHA1": None, "PHA2": None,
                          "MODE": 1, "JUMP": [False, 0],
                          "FORMAT": "Unknown", "END": False}
            self.observatories = set()
        with open(filename, "r") as f:
            for l in f.readlines():
                MJD, d = parse_TOA_line(l, fmt=self.cdict["FORMAT"])
                if d["format"] == "Command":
                    cmd = d["Command"][0]
                    self.commands.append((d["Command"], self.ntoas))
                    if cmd == "SKIP":
                        self.cdict[cmd] = True
                        continue
                    elif cmd == "NOSKIP":
                        self.cdict["SKIP"] = False
                        continue
                    elif cmd == "END":
                        self.cdict[cmd] = True
                        break
                    elif cmd in ("TIME", "PHASE"):
                        self.cdict[cmd] += float(d["Command"][1])
                    elif cmd in ("EMIN", "EMAX", "EFAC", "EQUAD",\
                                 "PHA1", "PHA2", "FMIN", "FMAX"):
                        self.cdict[cmd] = float(d["Command"][1])
                        if cmd in ("PHA1", "PHA2", "TIME", "PHASE"):
                            d[cmd] = d["Command"][1]
                    elif cmd == "INFO":
                        self.cdict[cmd] = d["Command"][1]
                        d[cmd] = d["Command"][1]
                    elif cmd == "FORMAT":
                        if d["Command"][1] == "1":
                            self.cdict[cmd] = "Tempo2"
                    elif cmd == "JUMP":
                        if self.cdict[cmd][0]:
                            self.cdict[cmd][0] = False
                            self.cdict[cmd][1] += 1
                        else:
                            self.cdict[cmd][0] = True
                    elif cmd == "INCLUDE" and process_includes:
                        # Save FORMAT in a tmp
                        fmt = self.cdict["FORMAT"]
                        self.cdict["FORMAT"] = "Unknown"
                        self.read_toa_file(d["Command"][1], top=False)
                        # re-set FORMAT
                        self.cdict["FORMAT"] = fmt
                    else:
                        continue
                if (self.cdict["SKIP"] or
                    d["format"] in ("Blank", "Unknown", "Comment", "Command")):
                    continue
                elif self.cdict["END"]:
                    if top:
                        # Clean up our temporaries used when reading TOAs
                        del self.cdict
                    return
                else:
                    newtoa = TOA(MJD, **d)
                    if ((self.cdict["EMIN"] > newtoa.error) or
                        (self.cdict["EMAX"] < newtoa.error) or
                        (self.cdict["FMIN"] > newtoa.freq) or
                        (self.cdict["FMAX"] < newtoa.freq)):
                        continue
                    else:
                        newtoa.error *= self.cdict["EFAC"]
                        newtoa.error = numpy.hypot(newtoa.error,
                                                   self.cdict["EQUAD"])
                        if self.cdict["INFO"]:
                            newtoa.flags["info"] = self.cdict["INFO"]
                        if self.cdict["JUMP"][0]:
                            newtoa.flags["jump"] = self.cdict["JUMP"][1]
                        if self.cdict["PHASE"] != 0:
                            newtoa.flags["phase"] = self.cdict["PHASE"]
                        if self.cdict["TIME"] != 0.0:
                            newtoa.flags["time"] = self.cdict["TIME"]
                        self.observatories.add(newtoa.obs)
                        self.toas.append(newtoa)
                        self.ntoas += 1
            if top:
                # Clean up our temporaries used when reading TOAs
                del self.cdict