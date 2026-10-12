{
  description = "Development environment for ttsbench-ja";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];
      perSystem = { pkgs, ... }: {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.ffmpeg
            pkgs.sox
            pkgs.uv
            pkgs.yt-dlp
          ];
          shellHook = ''
            ttsbench_python_site_packages="$PWD/.venv/lib/python3.10/site-packages"
            export LD_LIBRARY_PATH="$ttsbench_python_site_packages/nvidia/cublas/lib:$ttsbench_python_site_packages/nvidia/cudnn/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            unset ttsbench_python_site_packages
          '';
        };
      };
    };
}
