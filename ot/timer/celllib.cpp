#include <ot/timer/timer.hpp>

namespace ot {

// Function: _is_redundant_timing
bool Timer::_is_redundant_timing(const Timing& timing, Split el) const {

  switch(el) {
    case MIN:
      if(timing.is_max_constraint()) {
        return true; 
      }
    break;

    case MAX:
      if(timing.is_min_constraint()) {
        return true;
      }
    break;
  }

  return false;
}

// Function: read_celllib
Timer& Timer::read_celllib(std::filesystem::path path, std::optional<Split> el) {
  
  auto lib = std::make_shared<Celllib>();
  
  std::scoped_lock lock(_mutex);
  
  // Library parser
  auto parser = _taskflow.emplace([path=std::move(path), lib] () {
    OT_LOGI("loading celllib ", path);
    lib->read(path);
  });

  // Placeholder to add_lineage
  auto reader = _taskflow.emplace([this, lib, el] () {
    if(el) {
      _merge_celllib(*lib, *el);
    }
    else {
      auto cpy = *lib;
      _merge_celllib(cpy, MIN);
      _merge_celllib(*lib, MAX);
    }
  });

  // Reader -> reader
  parser.precede(reader);

  _add_to_lineage(reader);

  return *this;
}

// Procedure: _merge_celllib
void Timer::_merge_celllib(Celllib& lib, Split el) {

  _rebase_unit(lib);

  // initialize a library
  if(!_celllib[el]) {
    _celllib[el] = std::move(lib);
    OT_LOGI(
      "added ", to_string(el), " celllib ", std::quoted(_celllib[el]->name), 
      " [cells:", _celllib[el]->cells.size(), ']'
    );
  }
  // merge the library
  else {
    // Merge the lut template
    _celllib[el]->lut_templates.merge(std::move(lib.lut_templates));
    
    // Merge the cell
    _celllib[el]->cells.merge(std::move(lib.cells)); 
    
    OT_LOGI(
      "merged with library ", std::quoted(lib.name), 
      " [cells:", _celllib[el]->cells.size(), ']'
    );
  }
}

// https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
template<typename ... Args>
static std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

// Function: read_celllib
Timer& Timer::read_noise_models(std::filesystem::path path) {
  std::scoped_lock lock(_mutex);
  
  auto reader = _taskflow.emplace([this, path=std::move(path)] () {
    OT_LOGI("loading noise models under directory ", path);
    int cnt = 0;
    for(const auto &entry: std::filesystem::directory_iterator(path)) {
      const auto p = entry.path();
      _noise_models[p.filename()] = torch::jit::load(p.c_str());
      ++cnt;
    }
    OT_LOGI("loaded ", cnt, " noise models");

    // link the models into celllib Timing
    int cnt_links = 0;
    FOR_EACH_EL_IF(el, _celllib[el]) {
      for(auto &cell: _celllib[el]->cells) {
        for(auto &cp: cell.second.cellpins) {
          for(auto &tm: cp.second.timings) {
            FOR_EACH_RF(rf) {
              const std::string kw = string_format("%s_%s2%s_%s", cell.second.name.c_str(), tm.related_pin.c_str(), cp.second.name.c_str(), rf == RISE ? "rise" : "fall");
              tm.noise_model_kw = kw;
              // OT_LOGW(kw + "_tp_matrix_best.ptc");
              if(auto it = _noise_models.find(kw + "_tp_matrix_best.ptc"); it != _noise_models.end()) {
                tm.noise_model_tp[rf].emplace(&it->second);
                ++cnt_links;
              }
              if(auto it = _noise_models.find(kw + "_output_trans_matrix_best.ptc"); it != _noise_models.end()) {
                tm.noise_model_trans[rf].emplace(&it->second);
                ++cnt_links;
              }
            }
          }
        }
      }
    }
    OT_LOGI("linked the noise models to ", cnt_links, " library timings");
  });
  _add_to_lineage(reader);

  return *this;
}

};  // end of namespace ot. -----------------------------------------------------------------------




