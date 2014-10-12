# solar_system_shapiro.py
# Add in Shapiro delays due to solar system objects
import numpy
import astropy.units as u
import astropy.constants as const
from .parameter import Parameter
from .timing_model import TimingModel
from .. import Tsun, Tmercury, Tvenus, Tearth, Tmars, \
        Tjupiter, Tsaturn, Turanus, Tneptune

class SolarSystemShapiro(TimingModel):

    def __init__(self):
        super(SolarSystemShapiro, self).__init__()

        self.add_param(Parameter(name="PLANET_SHAPIRO",
            units=None, value=True, continuous=False,
            description="Include planetary Shapiro delays (Y/N)",
            parse_value=lambda x: x.upper() == 'Y',
            print_value=lambda x: 'Y' if x else 'N'))

        self.delay_funcs += [self.solar_system_shapiro_delay,]
        self.delay_funcs_ld += [self.solar_system_shapiro_delay_ld,]
        self.delay_funcs_table += [self.solar_system_shapiro_delay_table,]

    def setup(self):
        super(SolarSystemShapiro, self).setup()

    # Put masses in a convenient dictionary
    _ss_mass_sec = {"sun": Tsun.value,
                   "mercury": Tmercury.value,
                   "venus": Tvenus.value,
                   "earth": Tearth.value,
                   "mars": Tmars.value,
                   "jupiter": Tjupiter.value,
                   "saturn": Tsaturn.value,
                   "uranus": Turanus.value,
                   "neptune": Tneptune.value}

    @staticmethod
    def ss_obj_shapiro_delay(obj_pos, psr_dir, T_obj):
        """
        ss_obj_shapiro_delay(obj_pos, psr_dir, T_obj)

        returns Shapiro delay in seconds for a solar system object.

        Inputs:
          obj_pos : position vector from Earth to SS object, with Units
          psr_dir : unit vector in direction of pulsar
          T_obj : mass of object in seconds (GM/c^3)
        """
        r = numpy.sqrt(obj_pos.dot(obj_pos))
        rcostheta = obj_pos.dot(psr_dir)
        # This formula copied from tempo2 code.  The sign of the
        # cos(theta) term has been changed since we are using the
        # opposite convention for object position vector (from
        # observatory to object in this code).
        return -2.0 * T_obj * numpy.log((r-rcostheta)/const.au).value

    @staticmethod
    def ss_obj_shapiro_delay_array(obj_pos, psr_dir, T_obj):
        """
        An array version of ss_obj_shapiro_delay()
        obj_pos : position vector from Earth to SS object, array
        psr_dir : unit vector in direction of pulsar, array
        T_obj : mass of object in seconds (GM/c^3)
        """
        r = (numpy.sqrt(numpy.sum(obj_pos**2, axis = 1))) * u.km
        rcostheta = numpy.diag((numpy.dot(obj_pos, psr_dir))) * u.km
        return -2.0 * T_obj * numpy.log((r-rcostheta)/const.au).value  
    
    def solar_system_shapiro_delay(self, toa):
        """
        Returns total shapiro delay to due solar system objects.
        If the PLANET_SHAPIRO model param is set to True then
        planets are included, otherwise only the value for the
        Sun is calculated.

        Requires Astrometry or similar model that provides the
        ssb_to_psb_xyz method for direction to pulsar.

        If planets are to be included, TOAs.compute_posvels() must
        have been called with the planets=True argument.
        """
        psr_dir = self.ssb_to_psb_xyz(epoch=toa['mjd'].mjd)
        # Sun
        delay = self.ss_obj_shapiro_delay(toa['obs_sun_pvs'].pos,
                                          psr_dir,
                                          self._ss_mass_sec['sun'])
        if self.PLANET_SHAPIRO.value:
            for pl in ('jupiter', 'saturn', 'venus', 'uranus'):
                delay += self.ss_obj_shapiro_delay(
                    getattr(toa, 'obs_'+pl+'_pvs').pos,
                    psr_dir,
                    self._ss_mass_sec[pl])
        return delay

    def solar_system_shapiro_delay_ld(self, TOAs):
        """
        long double version of solar_system_shapiro_delay_ld
        """
        psr_dir = self.ssb_to_psb_xyzld(epoch = TOAs.tdbld)
        delay = numpy.zeros_like(TOAs.tdbld)
        for ii in range(len(TOAs.tdbld)):
            #SUN
            delay[ii] += self.ss_obj_shapiro_delay(TOAs.obs_sun_pvs[ii].pos,
                                                    psr_dir.value[:,ii],
                                                    self._ss_mass_sec['sun'])
            if self.PLANET_SHAPIRO.value:
                for pl in ('jupiter', 'saturn', 'venus', 'uranus'):
                    delay[ii] += self.ss_obj_shapiro_delay(
                        getattr(TOAs, 'obs_'+pl+'_pvs_ld')[ii].pos,
                        psr_dir.value[:,ii],
                        self._ss_mass_sec[pl])
        return delay

    def solar_system_shapiro_delay_table(self, TOAs):
        """
        Table version of solar_system_shapiro_delay
        """
        psr_dir = self.ssb_to_psb_xyzld(epoch = TOAs.dataTable['tdb_ld'])
        delay = numpy.zeros_like(TOAs.dataTable['tdb_ld'])
        # SUN
        delay += self.ss_obj_shapiro_delay_array(TOAs.dataTable['sun_posvel'][:,0:3],
                                                 psr_dir,
                                                 self._ss_mass_sec['sun']) 
        if self.PLANET_SHAPIRO.value:
            for pl in ('jupiter', 'saturn', 'venus', 'uranus'):
                delay += self.ss_obj_shapiro_delay_array(
                         TOAs.dataTable[pl+'_posvel'][:,0:3],
                         psr_dir,
                         self._ss_mass_sec[pl])
        return delay






        
        
        