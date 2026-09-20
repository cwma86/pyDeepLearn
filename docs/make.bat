@ECHO OFF

REM Minimal make.bat for Sphinx documentation.
REM
REM The canonical entry point for this project is the "docs" target in the
REM repository root makefile, which uses uv to run Sphinx:
REM
REM     make docs
REM
REM This file exists so that docs/ can also be built on its own.

pushd %~dp0

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help
if "%1" == "clean" goto clean
if "%1" == "html" goto html

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR%
goto end

:clean
if exist %BUILDDIR% rmdir /s /q %BUILDDIR%
goto end

:html
%SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%\html
echo.
echo Build finished. The HTML pages are in %BUILDDIR%\html.
goto end

:end
popd
